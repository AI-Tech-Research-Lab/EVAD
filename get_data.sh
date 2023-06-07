#/bin/bash

curl https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fm6xzxnf36-2.zip --output data.zip
mkdir data
unzip data.zip

mv fm6xzxnf36-2 TriaxalBearings
mv TriaxalBearings/ data

rm data.zip