#将env.tar.gz的文件放在~目录下

mkdir -p ../miniconda3 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ../miniconda3/miniconda.sh

bash ../miniconda3/miniconda.sh -b -u -p ../miniconda3

rm ../miniconda3/miniconda.sh

source ../miniconda3/bin/activate

conda init --all

conda create -n myenv-3.9 python=3.9

chmod -R 777 ../miniconda3/envs/myenv-3.9/

tar -zxvf ../envs.tar.gz

sudo rsync -avh --progress ./myenv-3.9/ ../miniconda3/envs/myenv-3.9/

rm -rf myenv-3.9

find ../miniconda3/envs/myenv-3.9/bin/ -type f -executable -print0 | while IFS= read -r -d '' file; do
  sed -i 's|/home/derder/miniconda3/envs/myenv-3.9/bin/python|../miniconda3/envs/myenv-3.9/bin/python|g' "$file"; 
done

conda activate myenv-3.9

# 检查该虚拟环境下是否有需要的所有库
pip list

# 如果你最后想把环境文件和资源文件的两个压缩包也删除的话，可以执行以下命令
# rm ../envs.tar.gz
# rm ../multimodel_rumor_detection.zip
