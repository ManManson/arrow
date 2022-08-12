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

#include <arrow/status.h>
#include <arrow/testing/random.h>
#include <arrow/util/async_generator.h>
#include <arrow/util/string_view.h>
#include <memory>

#include "benchmark/benchmark.h"

#include "arrow/filesystem/localfs.h"
#include "arrow/table.h"
#include "arrow/testing/future_util.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"
#include "arrow/util/formatting.h"
#include "arrow/util/io_util.h"
#include "arrow/util/make_unique.h"
#include "arrow/util/string_view.h"

#include "parquet/arrow/writer.h"
#include "parquet/properties.h"

#include <iostream>

namespace arrow {

namespace fs {

using arrow::internal::make_unique;
using arrow::internal::TemporaryDir;

/// Set up files and directories structure for benchmarking.
///
/// TODO: Describe topological structure of created directories and parquet files
class LocalFSFixture : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    ASSERT_OK_AND_ASSIGN(tmp_dir_, TemporaryDir::Make("localfs-test-"));

    auto options = LocalFileSystemOptions::Defaults();
    fs_ = make_unique<LocalFileSystem>(options);

    InitializeDatasetStructure(0, tmp_dir_->path());
  }

  void InitializeDatasetStructure(size_t cur_nesting_level,
                                  arrow::internal::PlatformFilename cur_root_dir) {
    // 1. Create `num_files_` under root_dir (Call MakeTrivialParquetFile(path))
    // 2. Create `num_dirs_` under current root_dir
    // 3. Call `InitializeDatasetStructure` on each one with (cur_nesting_level + 1)
    //    and (root_dir = current_root_dir / dir_name)
    ASSERT_OK(arrow::internal::CreateDir(cur_root_dir));

    arrow::internal::StringFormatter<Int32Type> format;

    for (size_t i = 0; i < num_files_; ++i) {
      std::string fname = "file_";
      format(i, [&fname](util::string_view formatted) {
        fname.append(formatted.data(), formatted.size());
      });
      fname.append(".parquet");
      ASSERT_OK_AND_ASSIGN(auto path, cur_root_dir.Join(std::move(fname)));
      ASSERT_OK(MakeTrivialParquetFile(path.ToString()));
    }

    if (cur_nesting_level == nesting_depth_) {
      return;
    }

    for (size_t i = 0; i < num_dirs_; ++i) {
      std::string dirname = "dir_";
      format(i, [&dirname](util::string_view formatted) {
        dirname.append(formatted.data(), formatted.size());
      });
      ASSERT_OK_AND_ASSIGN(auto path, cur_root_dir.Join(std::move(dirname)));
      InitializeDatasetStructure(cur_nesting_level + 1, std::move(path));
    }
  }

  Status MakeTrivialParquetFile(const std::string& path) {
    FieldVector fields{field("val", int64())};
    auto batch = random::GenerateBatch(fields, 1 /*num_rows*/, 0);
    ARROW_ASSIGN_OR_RAISE(auto table, Table::FromRecordBatches({batch}));

    std::shared_ptr<io::OutputStream> sink;
    ARROW_ASSIGN_OR_RAISE(sink, fs_->OpenOutputStream(path));

    RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink,
                                             1 /*num_rows*/));

    return Status::OK();
  }

 protected:
  std::unique_ptr<TemporaryDir> tmp_dir_;
  std::unique_ptr<LocalFileSystem> fs_;

  size_t nesting_depth_ = 2;
  size_t num_dirs_ = 10;
  size_t num_files_ = 1000;  // 10000
};

BENCHMARK_DEFINE_F(LocalFSFixture, AsyncFileDiscovery)
(benchmark::State& st) {
  for (auto _ : st) {
    auto options = LocalFileSystemOptions::Defaults();
    options.directory_readahead = st.range(0);
    options.file_info_batch_size = st.range(1);
    auto test_fs = make_unique<LocalFileSystem>(options);

    FileSelector select;
    select.base_dir = tmp_dir_->path().ToString();
    select.recursive = true;
    auto file_gen = test_fs->GetFileInfoGenerator(std::move(select));
    size_t total_file_count = 0;
    auto visit_fut =
        VisitAsyncGenerator(file_gen, [&total_file_count](const FileInfoVector& fv) {
          total_file_count += fv.size();
          return Status::OK();
        });
    ASSERT_FINISHES_OK(visit_fut);
    st.SetItemsProcessed(total_file_count);
  }
}
BENCHMARK_REGISTER_F(LocalFSFixture, AsyncFileDiscovery)
    // arg0: directory_readahead
    // arg1: file_info_batch_size
    ->ArgNames({"directory_readahead", "file_info_batch_size"})
    ->ArgsProduct({{1, 2, 4, 8, 16}, {1, 10, 100, 1000}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace fs

}  // namespace arrow
