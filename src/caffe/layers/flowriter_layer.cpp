// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"

#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>
#include <sys/dir.h>

#include "caffe/layers/flowriter_layer.hpp"
#include <string>


using std::max;

namespace caffe {

template <typename Dtype>
void FLOWriterLayer<Dtype>::writeFloFile(string filename, const float* data, int xSize, int ySize)
{
    FILE *stream = fopen(filename.c_str(), "wb");

    // write the header
    fprintf(stream,"PIEH");
    fwrite(&xSize,sizeof(int),1,stream);
    fwrite(&ySize,sizeof(int),1,stream);

    // write the data
    for (int y = 0; y < ySize; y++)
        for (int x = 0; x < xSize; x++) {
            float u = data[y*xSize+x];
            float v = data[y*xSize+x+ySize*xSize];
            fwrite(&u,sizeof(float),1,stream);
            fwrite(&v,sizeof(float),1,stream);
        }
    fclose(stream);
}
  
template <typename Dtype>
void FLOWriterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

  string root_folder = this->layer_param_.image_data_param().root_folder();

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  while (infile >> filename) {
    lines_.push_back(std::make_pair(filename, 0));
  }

  lines_id_ = 0;

  folder=this->layer_param_.writer_param().folder();

}

template <typename Dtype>
void FLOWriterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(bottom.size(), 1) << "FLOWRITER layer takes one input";

    const int channels = bottom[0]->channels();

    CHECK_EQ(channels, 2) << "FLOWRITER layer input must have two channels";

    DIR* dir = opendir(this->layer_param_.writer_param().folder().c_str());
    if (dir)
        closedir(dir);
    else if (ENOENT == errno) {
        std::string cmd("mkdir -p " + this->layer_param_.writer_param().folder());
        int retval = std::system(cmd.c_str());
        (void)retval;
    }
}

template <typename Dtype>
void FLOWriterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

 

    int size=height*width*channels;
    for(int n=0; n<num; n++)
    {
        string filename;
        
            
                if(lines_id_<lines_.size()){
                     
                string tmp=lines_[lines_id_].first;
                lines_id_++;
                int pos=tmp.find('.');
                filename=folder+"/"+tmp.substr(pos-9,9)+".flo";
               }
        

        const Dtype* data=bottom[0]->cpu_data()+n*size;

        LOG(INFO) << "Saving " << filename;
        writeFloFile(filename.c_str(),(const float*)data,width,height);
    }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(FLOWriterLayer, Forward);
#endif

INSTANTIATE_CLASS(FLOWriterLayer);
REGISTER_LAYER_CLASS(FLOWriter);

}  // namespace caffe
