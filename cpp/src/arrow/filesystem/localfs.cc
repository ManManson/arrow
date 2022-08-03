// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/type_fwd.h>
#include <arrow/io/type_fwd.h>
#include <arrow/status.h>
#include <arrow/util/async_generator.h>
#include <chrono>
#include <cstring>
#include <sstream>
#include <utility>

#ifdef _WIN32
#include "arrow/util/windows_compatibility.h"
#else
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#endif

#include "arrow/dataset/discovery.h"
#include "arrow/filesystem/localfs.h"
#include "arrow/filesystem/path_util.h"
#include "arrow/filesystem/util_internal.h"
#include "arrow/io/file.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"
#include "arrow/util/uri.h"
#include "arrow/util/windows_fixup.h"

#include <thread>

namespace arrow {

template <>
struct IterationTraits<fs::FileInfo> {
  static fs::FileInfo End() { return {}; }
  static bool IsEnd(const fs::FileInfo& val) { return val.Equals(End()); }
};

namespace fs {
using ::arrow::internal::IOErrorFromErrno;
#ifdef _WIN32
using ::arrow::internal::IOErrorFromWinError;
#endif
using ::arrow::internal::NativePathString;
using ::arrow::internal::PlatformFilename;

namespace internal {

#ifdef _WIN32
static bool IsDriveLetter(char c) {
  // Can't use locale-dependent functions from the C/C++ stdlib
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
#endif

bool DetectAbsolutePath(const std::string& s) {
  // Is it a /-prefixed local path?
  if (s.length() >= 1 && s[0] == '/') {
    return true;
  }
#ifdef _WIN32
  // Is it a \-prefixed local path?
  if (s.length() >= 1 && s[0] == '\\') {
    return true;
  }
  // Does it start with a drive letter in addition to being /- or \-prefixed,
  // e.g. "C:\..."?
  if (s.length() >= 3 && s[1] == ':' && (s[2] == '/' || s[2] == '\\') &&
      IsDriveLetter(s[0])) {
    return true;
  }
#endif
  return false;
}

}  // namespace internal

namespace {

Status ValidatePath(util::string_view s) {
  if (internal::IsLikelyUri(s)) {
    return Status::Invalid("Expected a local filesystem path, got a URI: '", s, "'");
  }
  return Status::OK();
}

#ifdef _WIN32

std::string NativeToString(const NativePathString& ns) {
  PlatformFilename fn(ns);
  return fn.ToString();
}

TimePoint ToTimePoint(FILETIME ft) {
  // Hundreds of nanoseconds between January 1, 1601 (UTC) and the Unix epoch.
  static constexpr int64_t kFileTimeEpoch = 11644473600LL * 10000000;

  int64_t hundreds = (static_cast<int64_t>(ft.dwHighDateTime) << 32) + ft.dwLowDateTime -
                     kFileTimeEpoch;  // hundreds of ns since Unix epoch
  std::chrono::nanoseconds ns_count(100 * hundreds);
  return TimePoint(std::chrono::duration_cast<TimePoint::duration>(ns_count));
}

FileInfo FileInformationToFileInfo(const BY_HANDLE_FILE_INFORMATION& information) {
  FileInfo info;
  if (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
    info.set_type(FileType::Directory);
    info.set_size(kNoSize);
  } else {
    // Regular file
    info.set_type(FileType::File);
    info.set_size((static_cast<int64_t>(information.nFileSizeHigh) << 32) +
                  information.nFileSizeLow);
  }
  info.set_mtime(ToTimePoint(information.ftLastWriteTime));
  return info;
}

Result<FileInfo> StatFile(const std::wstring& path) {
  HANDLE h;
  std::string bytes_path = NativeToString(path);
  FileInfo info;

  /* Inspired by CPython, see Modules/posixmodule.c */
  h = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, /* desired access */
                  0,                                  /* share mode */
                  NULL,                               /* security attributes */
                  OPEN_EXISTING,
                  /* FILE_FLAG_BACKUP_SEMANTICS is required to open a directory */
                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_BACKUP_SEMANTICS, NULL);

  if (h == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      info.set_path(bytes_path);
      info.set_type(FileType::NotFound);
      info.set_mtime(kNoTime);
      info.set_size(kNoSize);
      return info;
    } else {
      return IOErrorFromWinError(GetLastError(), "Failed querying information for path '",
                                 bytes_path, "'");
    }
  }
  BY_HANDLE_FILE_INFORMATION information;
  if (!GetFileInformationByHandle(h, &information)) {
    CloseHandle(h);
    return IOErrorFromWinError(GetLastError(), "Failed querying information for path '",
                               bytes_path, "'");
  }
  CloseHandle(h);
  info = FileInformationToFileInfo(information);
  info.set_path(bytes_path);
  return info;
}

#else  // POSIX systems

TimePoint ToTimePoint(const struct timespec& s) {
  std::chrono::nanoseconds ns_count(static_cast<int64_t>(s.tv_sec) * 1000000000 +
                                    static_cast<int64_t>(s.tv_nsec));
  return TimePoint(std::chrono::duration_cast<TimePoint::duration>(ns_count));
}

FileInfo StatToFileInfo(const struct stat& s) {
  FileInfo info;
  if (S_ISREG(s.st_mode)) {
    info.set_type(FileType::File);
    info.set_size(static_cast<int64_t>(s.st_size));
  } else if (S_ISDIR(s.st_mode)) {
    info.set_type(FileType::Directory);
    info.set_size(kNoSize);
  } else {
    info.set_type(FileType::Unknown);
    info.set_size(kNoSize);
  }
#ifdef __APPLE__
  // macOS doesn't use the POSIX-compliant spelling
  info.set_mtime(ToTimePoint(s.st_mtimespec));
#else
  info.set_mtime(ToTimePoint(s.st_mtim));
#endif
  return info;
}

Result<FileInfo> StatFile(const std::string& path) {
  FileInfo info;
  struct stat s;
  int r = stat(path.c_str(), &s);
  if (r == -1) {
    if (errno == ENOENT || errno == ENOTDIR || errno == ELOOP) {
      info.set_type(FileType::NotFound);
      info.set_mtime(kNoTime);
      info.set_size(kNoSize);
    } else {
      return IOErrorFromErrno(errno, "Failed stat()ing path '", path, "'");
    }
  } else {
    info = StatToFileInfo(s);
  }
  info.set_path(path);
  return info;
}

#endif

Status StatSelector(const PlatformFilename& dir_fn, const FileSelector& select,
                    int32_t nesting_depth, std::vector<FileInfo>* out) {
  auto result = ListDir(dir_fn);
  if (!result.ok()) {
    auto status = result.status();
    if (select.allow_not_found && status.IsIOError()) {
      ARROW_ASSIGN_OR_RAISE(bool exists, FileExists(dir_fn));
      if (!exists) {
        return Status::OK();
      }
    }
    return status;
  }

  for (const auto& child_fn : *result) {
    PlatformFilename full_fn = dir_fn.Join(child_fn);
    ARROW_ASSIGN_OR_RAISE(FileInfo info, StatFile(full_fn.ToNative()));
    if (info.type() != FileType::NotFound) {
      out->push_back(std::move(info));
    }
    if (nesting_depth < select.max_recursion && select.recursive &&
        info.type() == FileType::Directory) {
      RETURN_NOT_OK(StatSelector(full_fn, select, nesting_depth + 1, out));
    }
  }
  return Status::OK();
}

}  // namespace

LocalFileSystemOptions LocalFileSystemOptions::Defaults() {
  return LocalFileSystemOptions();
}

bool LocalFileSystemOptions::Equals(const LocalFileSystemOptions& other) const {
  return use_mmap == other.use_mmap;
}

Result<LocalFileSystemOptions> LocalFileSystemOptions::FromUri(
    const ::arrow::internal::Uri& uri, std::string* out_path) {
  if (!uri.username().empty() || !uri.password().empty()) {
    return Status::Invalid("Unsupported username or password in local URI: '",
                           uri.ToString(), "'");
  }
  std::string path;
  const auto host = uri.host();
  if (!host.empty()) {
#ifdef _WIN32
    std::stringstream ss;
    ss << "//" << host << "/" << internal::RemoveLeadingSlash(uri.path());
    *out_path = ss.str();
#else
    return Status::Invalid("Unsupported hostname in non-Windows local URI: '",
                           uri.ToString(), "'");
#endif
  } else {
    *out_path = uri.path();
  }

  // TODO handle use_mmap option
  return LocalFileSystemOptions();
}

LocalFileSystem::LocalFileSystem(const io::IOContext& io_context)
    : FileSystem(io_context), options_(LocalFileSystemOptions::Defaults()) {}

LocalFileSystem::LocalFileSystem(const LocalFileSystemOptions& options,
                                 const io::IOContext& io_context)
    : FileSystem(io_context), options_(options) {}

LocalFileSystem::~LocalFileSystem() {}

Result<std::string> LocalFileSystem::NormalizePath(std::string path) {
  RETURN_NOT_OK(ValidatePath(path));
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(path));
  return fn.ToString();
}

bool LocalFileSystem::Equals(const FileSystem& other) const {
  if (other.type_name() != type_name()) {
    return false;
  } else {
    const auto& localfs = ::arrow::internal::checked_cast<const LocalFileSystem&>(other);
    return options_.Equals(localfs.options());
  }
}

Result<FileInfo> LocalFileSystem::GetFileInfo(const std::string& path) {
  RETURN_NOT_OK(ValidatePath(path));
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(path));
  return StatFile(fn.ToNative());
}

namespace {

bool StartsWithAnyOf(const std::string& path, const std::vector<std::string>& prefixes) {
  if (prefixes.empty()) {
    return false;
  }

  auto parts = fs::internal::SplitAbstractPath(path);
  return std::any_of(parts.cbegin(), parts.cend(), [&](util::string_view part) {
    return std::any_of(prefixes.cbegin(), prefixes.cend(), [&](util::string_view prefix) {
      return util::string_view(part).starts_with(prefix);
    });
  });
}

struct StatOptions {
  /// Invalid files (via selector or explicitly) will be excluded by checking
  /// with the FileFormat::IsSupported method.  This will incur IO for each files
  /// in a serial and single threaded fashion. Disabling this feature will skip the
  /// IO, but unsupported files may be present in the Dataset
  /// (resulting in an error at scan time).
  bool exclude_invalid_files = false;

  /// When discovering from a Selector (and not from an explicit file list), ignore
  /// files and directories matching any of these prefixes.
  ///
  /// Example (with selector = "/dataset/**"):
  /// selector_ignore_prefixes = {"_", ".DS_STORE" };
  ///
  /// - "/dataset/data.csv" -> not ignored
  /// - "/dataset/_metadata" -> ignored
  /// - "/dataset/.DS_STORE" -> ignored
  /// - "/dataset/_hidden/dat" -> ignored
  /// - "/dataset/nested/.DS_STORE" -> ignored
  std::vector<std::string> selector_ignore_prefixes = {
      ".",
      "_",
  };
  /// How many partitions should be processed in parallel.
  int32_t partitions_readahead = 1;
};

using SinglePartitionGenerator = AsyncGenerator<FileInfoVector>;
using SinglePartitionPushGenerator = PushGenerator<SinglePartitionGenerator>;
using SinglePartitionProducer = SinglePartitionPushGenerator::Producer;
using PartitionsGenerator = AsyncGenerator<SinglePartitionGenerator>;

class AsyncStatSelector {
 public:
  using FilterFn = std::function<Result<bool>(const FileInfo&)>;

  static Result<PartitionsGenerator> DiscoverPartitions(FileSelector selector,
                                                        const StatOptions& opts) {
    PushGenerator<SinglePartitionGenerator> file_gen;

    auto filter_fn = [selector, &opts](const FileInfo& info) {
      return FileFilter(info, selector, opts);
    };

    ARROW_ASSIGN_OR_RAISE(
        auto base_dir, arrow::internal::PlatformFilename::FromString(selector.base_dir));
    ARROW_RETURN_NOT_OK(PerformDiscovery(std::move(base_dir), 0, std::move(filter_fn),
                                         std::move(selector), file_gen.producer()));

    return file_gen;
  }

  static arrow::Result<SinglePartitionGenerator> DiscoverPartitionsFlattened(
      FileSelector selector, const StatOptions& opts) {
    ARROW_ASSIGN_OR_RAISE(auto part_gen, DiscoverPartitions(std::move(selector), opts));
    return opts.partitions_readahead > 1
               ? MakeSequencedMergedGenerator(std::move(part_gen),
                                              opts.partitions_readahead)
               : MakeConcatenatedGenerator(std::move(part_gen));
  }

 private:
  class DiscoveryImplIterator {
    static constexpr size_t batch_size = 1000;

    PlatformFilename dir_fn_;
    int32_t nesting_depth_;
    bool initialized_ = false;
    std::vector<PlatformFilename> child_fns_;
    size_t idx_ = 0;

    FilterFn filter_;

    SinglePartitionProducer partition_producer_;
    FileSelector selector_;

    FileInfoVector current_chunk_;

   public:
    DiscoveryImplIterator(PlatformFilename dir_fn, int32_t nesting_depth, FilterFn filter,
                          FileSelector selector,
                          SinglePartitionProducer partition_producer)
        : dir_fn_(std::move(dir_fn)),
          nesting_depth_(nesting_depth),
          filter_(std::move(filter)),
          partition_producer_(std::move(partition_producer)),
          selector_(std::move(selector)) {
      current_chunk_.reserve(batch_size);
    }

    Status Initialize() {
      auto result = arrow::internal::ListDir(dir_fn_);
      if (!result.ok()) {
        auto status = result.status();
        if (selector_.allow_not_found && status.IsIOError()) {
          auto exists = FileExists(dir_fn_);
          if (exists.ok() && !*exists) {
            return Status::OK();
          } else {
            return exists.ok() ? arrow::Status::UnknownError(
                                     "Failed to discover directory: ", dir_fn_.ToNative())
                               : exists.status();
          }
        }
        return status;
      }
      child_fns_ = std::move(result.MoveValueUnsafe());
      initialized_ = true;
      return Status::OK();
    }

    Result<FileInfoVector> Next() {
      if (!initialized_) {
        RETURN_NOT_OK(Initialize());
      }
      while (idx_ < child_fns_.size()) {
        auto full_fn = dir_fn_.Join(child_fns_[idx_++]);
        auto res = StatFile(full_fn.ToNative());
        if (!res.ok()) {
          return Finish(res.status());
        }

        auto info = res.MoveValueUnsafe();

        if (info.type() == FileType::Directory &&
            nesting_depth_ < selector_.max_recursion && selector_.recursive) {
          // вот здесь запускается в том же тредпуле еще один дискавери, точнее,
          // конструируется итератор в субдиректории и из него делается генератор
          // далее пушатся данные из этого генератора в общий синк.
          //
          // тут фишка в том, что opts является супер-аргументом, который не только
          // options передает, но также и сам синк, а его, по идее, надо бы
          // перекидывать
          // явно
          auto status = PerformDiscovery(full_fn, nesting_depth_ + 1, filter_, selector_,
                                         partition_producer_);
          if (!status.ok()) {
            return Finish(status);
          }
          continue;
        }

        auto check = filter_(info);
        if (!check.ok()) {
          return Finish(check.status());
        } else if (*check) {
          current_chunk_.emplace_back(std::move(info));
          if (current_chunk_.size() == batch_size) {
            auto yielded_vec = current_chunk_;
            current_chunk_.clear();
            return yielded_vec;
          }
        }
        continue;
      }
      if (!current_chunk_.empty()) {
        auto yielded_vec = current_chunk_;
        current_chunk_.clear();
        return yielded_vec;
      }
      return Finish();
    }

   private:
    Result<FileInfoVector> Finish(Status status = Status::OK()) {
      partition_producer_.Close();
      ARROW_RETURN_NOT_OK(status);
      return IterationEnd<FileInfoVector>();
    }
  };

  // Create a DiscoveryImplIterator under the hood, convert to a generator
  // feed it to the outer PushGenerator's producer queue (Discover(3))
  static Status PerformDiscovery(const PlatformFilename& dir_fn, int32_t nesting_depth,
                                 FilterFn filter, FileSelector selector,
                                 SinglePartitionProducer partition_producer) {
    ARROW_RETURN_IF(partition_producer.is_closed(),
                    arrow::Status::Cancelled("Discovery cancelled"));

    ARROW_ASSIGN_OR_RAISE(
        auto gen,
        MakeBackgroundGenerator(Iterator<FileInfoVector>(DiscoveryImplIterator(
                                    std::move(dir_fn), nesting_depth, std::move(filter),
                                    std::move(selector), partition_producer)),
                                io::default_io_context().executor()));
    gen = MakeTransferredGenerator(std::move(gen), arrow::internal::GetCpuThreadPool());
    ARROW_RETURN_IF(!partition_producer.Push(std::move(gen)),
                    arrow::Status::Cancelled("Discovery cancelled"));
    return arrow::Status::OK();
  }

  static Result<bool> FileFilter(const FileInfo& info, const FileSelector& selector,
                                 const StatOptions& opts) {
    if (!opts.exclude_invalid_files) {
      return true;
    }
    if (!info.IsFile()) {
      return false;
    };
    auto relative = fs::internal::RemoveAncestor(selector.base_dir, info.path());
    if (!relative) {
      return Status::Invalid("GetFileInfo() yielded path '", info.path(),
                             "', which is outside base dir '", selector.base_dir, "'");
    }
    if (StartsWithAnyOf(std::string(*relative), opts.selector_ignore_prefixes)) {
      return false;
    }
    return true;
  }
};

}  // anonymous namespace

Result<std::vector<FileInfo>> LocalFileSystem::GetFileInfo(const FileSelector& select) {
  RETURN_NOT_OK(ValidatePath(select.base_dir));
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(select.base_dir));
  std::vector<FileInfo> results;
  RETURN_NOT_OK(StatSelector(fn, select, 0, &results));
  return results;
}

FileInfoGenerator LocalFileSystem::GetFileInfoGenerator(const FileSelector& select) {
  auto path_status = ValidatePath(select.base_dir);
  if (!path_status.ok()) {
    return MakeFailingGenerator<FileInfoVector>(path_status);
  }
  auto fileinfo_gen =
      AsyncStatSelector::DiscoverPartitionsFlattened(select, StatOptions{});
  if (!fileinfo_gen.ok()) {
    return MakeFailingGenerator<FileInfoVector>(fileinfo_gen.status());
  }
  return fileinfo_gen.MoveValueUnsafe();
}

Status LocalFileSystem::CreateDir(const std::string& path, bool recursive) {
  RETURN_NOT_OK(ValidatePath(path));
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(path));
  if (recursive) {
    return ::arrow::internal::CreateDirTree(fn).status();
  } else {
    return ::arrow::internal::CreateDir(fn).status();
  }
}

Status LocalFileSystem::DeleteDir(const std::string& path) {
  RETURN_NOT_OK(ValidatePath(path));
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(path));
  auto st = ::arrow::internal::DeleteDirTree(fn, /*allow_not_found=*/false).status();
  if (!st.ok()) {
    // TODO Status::WithPrefix()?
    std::stringstream ss;
    ss << "Cannot delete directory '" << path << "': " << st.message();
    return st.WithMessage(ss.str());
  }
  return Status::OK();
}

Status LocalFileSystem::DeleteDirContents(const std::string& path, bool missing_dir_ok) {
  RETURN_NOT_OK(ValidatePath(path));
  if (internal::IsEmptyPath(path)) {
    return internal::InvalidDeleteDirContents(path);
  }
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(path));
  auto st = ::arrow::internal::DeleteDirContents(fn, missing_dir_ok).status();
  if (!st.ok()) {
    std::stringstream ss;
    ss << "Cannot delete directory contents in '" << path << "': " << st.message();
    return st.WithMessage(ss.str());
  }
  return Status::OK();
}

Status LocalFileSystem::DeleteRootDirContents() {
  return Status::Invalid("LocalFileSystem::DeleteRootDirContents is strictly forbidden");
}

Status LocalFileSystem::DeleteFile(const std::string& path) {
  RETURN_NOT_OK(ValidatePath(path));
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(path));
  return ::arrow::internal::DeleteFile(fn, /*allow_not_found=*/false).status();
}

Status LocalFileSystem::Move(const std::string& src, const std::string& dest) {
  RETURN_NOT_OK(ValidatePath(src));
  RETURN_NOT_OK(ValidatePath(dest));
  ARROW_ASSIGN_OR_RAISE(auto sfn, PlatformFilename::FromString(src));
  ARROW_ASSIGN_OR_RAISE(auto dfn, PlatformFilename::FromString(dest));

#ifdef _WIN32
  if (!MoveFileExW(sfn.ToNative().c_str(), dfn.ToNative().c_str(),
                   MOVEFILE_REPLACE_EXISTING)) {
    return IOErrorFromWinError(GetLastError(), "Failed renaming '", sfn.ToString(),
                               "' to '", dfn.ToString(), "'");
  }
#else
  if (rename(sfn.ToNative().c_str(), dfn.ToNative().c_str()) == -1) {
    return IOErrorFromErrno(errno, "Failed renaming '", sfn.ToString(), "' to '",
                            dfn.ToString(), "'");
  }
#endif
  return Status::OK();
}

Status LocalFileSystem::CopyFile(const std::string& src, const std::string& dest) {
  RETURN_NOT_OK(ValidatePath(src));
  RETURN_NOT_OK(ValidatePath(dest));
  ARROW_ASSIGN_OR_RAISE(auto sfn, PlatformFilename::FromString(src));
  ARROW_ASSIGN_OR_RAISE(auto dfn, PlatformFilename::FromString(dest));
  // XXX should we use fstat() to compare inodes?
  if (sfn.ToNative() == dfn.ToNative()) {
    return Status::OK();
  }

#ifdef _WIN32
  if (!CopyFileW(sfn.ToNative().c_str(), dfn.ToNative().c_str(),
                 FALSE /* bFailIfExists */)) {
    return IOErrorFromWinError(GetLastError(), "Failed copying '", sfn.ToString(),
                               "' to '", dfn.ToString(), "'");
  }
  return Status::OK();
#else
  ARROW_ASSIGN_OR_RAISE(auto is, OpenInputStream(src));
  ARROW_ASSIGN_OR_RAISE(auto os, OpenOutputStream(dest));
  RETURN_NOT_OK(internal::CopyStream(is, os, 1024 * 1024 /* chunk_size */, io_context()));
  RETURN_NOT_OK(os->Close());
  return is->Close();
#endif
}

namespace {

template <typename InputStreamType>
Result<std::shared_ptr<InputStreamType>> OpenInputStreamGeneric(
    const std::string& path, const LocalFileSystemOptions& options,
    const io::IOContext& io_context) {
  RETURN_NOT_OK(ValidatePath(path));
  if (options.use_mmap) {
    return io::MemoryMappedFile::Open(path, io::FileMode::READ);
  } else {
    return io::ReadableFile::Open(path, io_context.pool());
  }
}

}  // namespace

Result<std::shared_ptr<io::InputStream>> LocalFileSystem::OpenInputStream(
    const std::string& path) {
  return OpenInputStreamGeneric<io::InputStream>(path, options_, io_context());
}

Result<std::shared_ptr<io::RandomAccessFile>> LocalFileSystem::OpenInputFile(
    const std::string& path) {
  return OpenInputStreamGeneric<io::RandomAccessFile>(path, options_, io_context());
}

namespace {

Result<std::shared_ptr<io::OutputStream>> OpenOutputStreamGeneric(const std::string& path,
                                                                  bool truncate,
                                                                  bool append) {
  RETURN_NOT_OK(ValidatePath(path));
  ARROW_ASSIGN_OR_RAISE(auto fn, PlatformFilename::FromString(path));
  const bool write_only = true;
  ARROW_ASSIGN_OR_RAISE(
      auto fd, ::arrow::internal::FileOpenWritable(fn, write_only, truncate, append));
  int raw_fd = fd.Detach();
  auto maybe_stream = io::FileOutputStream::Open(raw_fd);
  if (!maybe_stream.ok()) {
    ARROW_UNUSED(::arrow::internal::FileClose(raw_fd));
  }
  return maybe_stream;
}

}  // namespace

Result<std::shared_ptr<io::OutputStream>> LocalFileSystem::OpenOutputStream(
    const std::string& path, const std::shared_ptr<const KeyValueMetadata>& metadata) {
  bool truncate = true;
  bool append = false;
  return OpenOutputStreamGeneric(path, truncate, append);
}

Result<std::shared_ptr<io::OutputStream>> LocalFileSystem::OpenAppendStream(
    const std::string& path, const std::shared_ptr<const KeyValueMetadata>& metadata) {
  bool truncate = false;
  bool append = true;
  return OpenOutputStreamGeneric(path, truncate, append);
}

}  // namespace fs
}  // namespace arrow
