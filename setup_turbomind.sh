# 检查 thrid_party/turbomind 是否存在，如果不存在提示需要git子项目初始化
if [ ! -d "third_party/turbomind" ]; then
    echo "Please run git submodule update --init --recursive first"
    exit 1
fi

cd third_party/turbomind

# 第一个入参 proc
proc=$1
if [ -z "$proc" ]; then
    proc=1
fi
cuda_version=$2
if [ -z "$cuda_version" ]; then
    cuda_version=12
fi

mkdir -p build && cd build && rm -rf *
bash ../generate.sh make
make -j${proc} && make install
if [ $? != 0 ]; then
    echo "build failed"
    exit 1
fi
cd ..
rm -rf build

mkdir -p ../../turbomind_build/
python3 setup.py bdist_wheel --cuda=${cuda_version} -d ../../turbomind_build/
chown ${USERID}:${GROUPID} ../../turbomind_build/*

cd ../../
pip uninstall turbomind -y
pip install ./turbomind_build/*.whl
