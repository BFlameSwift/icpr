
#!/bin/bash

# ����·��
TMP_DIR=../temp_data
SRC_DIR1=../data/testBdata/gray
ZIP_FILE=../data.zip

# ������ʱĿ¼�����������Ŀ¼�ṹ
mkdir -p $TMP_DIR/data
cp -r $SRC_DIR1 $TMP_DIR/data/


# ����ʱĿ¼�д��� ZIP �ļ�
cd $TMP_DIR
zip -r $ZIP_FILE data

# ����ԭʼĿ¼
cd -

# ɾ����ʱĿ¼
rm -rf $TMP_DIR

echo "Compression completed."