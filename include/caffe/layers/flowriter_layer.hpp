#ifndef CAFFE_FLOWRITER_LAYER_HPP
#define CAFFE_FLOWRITER_LAYER_HPP

#include <string>
#include <utility>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/layers/base_data_layer.hpp"

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"


namespace caffe {

/**
 * @brief FLOWriterLayer writes FLO (flow) files
 *
 */
template <typename Dtype>
class FLOWriterLayer : public Layer<Dtype> {
 public:
  explicit FLOWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~FLOWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "FLOWriter"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
 protected:

  void writeFloFile(string filename, const float* data, int xSize, int ySize);
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
  string folder;
};


}  // namespace caffe

#endif  // CAFFE_FLOWRITER_LAYER_HPP_
