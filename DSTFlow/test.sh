TOOLS=../build/tools


$TOOLS/caffe test --model=test_deploy.prototxt  --weights=./trained_model/dst_chair.caffemodel  -gpu 0 --iterations=640  
