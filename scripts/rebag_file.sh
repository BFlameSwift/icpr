#!/bin/bash

# 设置路径
TMP_DIR=../temp_data
SRC_DIR1=../data/test
SRC_DIR2=../data/train
ZIP_FILE=../data.zip

# 创建临时目录并复制所需的目录结构
mkdir -p $TMP_DIR/data
cp -r $SRC_DIR1 $TMP_DIR/data/
cp -r $SRC_DIR2 $TMP_DIR/data/

# 从临时目录中创建 ZIP 文件
cd $TMP_DIR
zip -r $ZIP_FILE data

# 返回原始目录
cd -

# 删除临时目录
rm -rf $TMP_DIR

echo "Compression completed."