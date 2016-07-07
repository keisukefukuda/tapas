#ifndef TAPAS_UTIL_H_
#define TAPAS_UTIL_H_

#include <cassert>
#include <cerrno>
#include <cstring>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#ifdef USE_MPI
# include <mpi.h>
#endif

namespace tapas {
namespace util {

void OpenFileStream(std::ofstream &ofs, const char *fname, decltype(std::ios::out) mode) {
  ofs.clear();

  int err_cnt = 0;

  while(1) {
    ofs.open(fname, mode);
    if (ofs.good()) {
      break;
    } else {
      if (err_cnt++ >= 3) {
        // fatal error. abort
        std::cerr << "Tapas: [FATAL] open failed ('" << fname << "'): " << strerror(errno) << std::endl;
        exit(-1);
      } else {
        std::cerr << "Tapas: [ERROR] open failed ('" << fname << "'): " << strerror(errno) << std::endl;
        usleep(5e5);
      }
    }
  }

  if (err_cnt > 0) {
    std::cerr << "Tapas: [WARNING] open('" << fname << "') failed " << err_cnt << " time"
              << (err_cnt > 1 ? "s" : "")
              << " but seems to be recovered." << std::endl;
  }
}

namespace {

template<class T>
std::string Format(T v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

std::string Format(double v) {
  std::stringstream ss;
  ss << std::scientific << v;
  return ss.str();
}

std::string Format(float v) {
  std::stringstream ss;
  ss << std::scientific << v;
  return ss.str();
}

} // anon namespace

struct ValueBase {
  virtual std::string get() const = 0;
  virtual ~ValueBase() { }
};

template <class T>
struct ValueImpl : public ValueBase {
  T value_;
  ValueImpl(T value) : value_(value) { }

  virtual std::string get() const override {
    return Format(value_);
  }
  virtual ~ValueImpl() {}
};

struct Value {
  ValueBase *value_;

  Value() : value_(nullptr) {}
  ~Value() { if (value_) delete value_; }

  template<class T>
  Value& operator=(T v) {
    //std::cout << "Value::operator=(" << v << ") is called." << std::endl;
    if (value_) {
      delete value_;
      value_ = nullptr;
    }
    value_ = new ValueImpl<T>(v);
    return *this;
  }

  std::string get() const {
    if (value_) return value_->get();
    else return std::string("N/A");
  }
};



class CSV {

  static const constexpr int kDefaultWidth = 14;
  
  const size_t ncols_;
  const size_t nrows_;
  size_t col_width_;

  const std::vector<std::string> cols_;

  std::unique_ptr<Value[]> data_;
  std::unordered_map<std::string, size_t> col2idx_;

 public:
  CSV(std::initializer_list<std::string> cols, size_t nrows)
      : ncols_(cols.size())
      , nrows_(nrows)
      , col_width_(kDefaultWidth)
      , cols_(cols)
      , data_(new Value[ncols_ * nrows_], std::default_delete<Value[]>())
      , col2idx_()
  {
    int i = 0;
    for (auto &&col_name : cols) {
      col2idx_[col_name] = i++;
    }
  }
  CSV(std::vector<std::string> cols, size_t nrows)
      : ncols_(cols.size())
      , nrows_(nrows)
      , col_width_(kDefaultWidth)
      , cols_(cols)
      , data_(new Value[ncols_ * nrows_], std::default_delete<Value[]>())
      , col2idx_()
  {
    int i = 0;
    for (auto &&col_name : cols) {
      col2idx_[col_name] = i++;
    }
  }

  void SetColumnWidth(size_t w) {
    assert(w > 0);
    col_width_ = w;
  }
  
  Value &At(const std::string &col, size_t irow) {
    if (col2idx_.count(col) == 0) {
      std::cerr << "ERROR: Unknown column name: " << col << std::endl;
      exit(-1);
    }
    size_t icol = col2idx_[col];
    return data_.get()[icol * nrows_ + irow];
  }

  void DumpHeader(std::ostream &os) const {
    for (auto &&col : cols_) {
      int npad = col_width_ - col.size();
      if (npad > 0) {
        for (int i = 0; i < npad; i++) {
          os << " ";
        }
      }
      os << col;
    }
    os << std::endl;
  }

  void Dump(std::ostream &os, bool header = true) const {
    if (header) {
      DumpHeader(os);
    }
    
    for (size_t row = 0; row < nrows_; row++) {
      for (size_t col = 0; col < ncols_; col++) {
        std::string v = data_.get()[col * nrows_ + row].get();

        int npad = col_width_ - v.size();

        // padding before the value
        for (int i = 0; i < npad; i++) {
          os << " ";
        }
        os << v;

        // padding after the value (if npad <= 0, which means the value is equal or longer than column width.
        // we need at least one padding.
        if (col < ncols_- 1 && npad <= 0) {
          os << " ";
        }
      }
      os << std::endl;
    }
  }

  void Dump(const char *fname) const {
    std::ofstream ofs;
    OpenFileStream(ofs, fname, std::ios::out);
    Dump(ofs);
    ofs.close();
  }

  void Dump(const std::string &fname) const {
    Dump(fname.c_str());
  }
};

#ifdef USE_MPI

class RankCSV {
  std::unique_ptr<CSV> csv_;
  int mpi_rank_;
  MPI_Comm comm_;
  
 public:
  RankCSV(std::initializer_list<std::string> cols)
  {
    std::vector<std::string> cols2 = cols;
    cols2.insert(cols2.begin(), "Rank");
    csv_.reset(new CSV(cols2, 1));
    comm_ = MPI_COMM_WORLD;
    MPI_Comm_rank(comm_, &mpi_rank_);

    this->At("Rank") = mpi_rank_;
  }
  
  Value &At(const std::string &col) {
    return csv_->At(col, 0);
  }

  void Dump(const std::string &fname) const {
    Dump(fname.c_str());
  }
  
  void Dump(const char *fname) const {
    int mpi_size;
    MPI_Comm_size(comm_, &mpi_size);
    
    std::stringstream ss;
    csv_->Dump(ss, false);
    std::string my_row = ss.str();

    int len = my_row.size() + 1;
    int max_len = 0;

    MPI_Allreduce(&len, &max_len, 1, MPI_INT, MPI_MAX, comm_);

    char *recv_buf = nullptr;

    if (mpi_rank_ == 0) {
      recv_buf = new char[max_len * mpi_size];
    }
    
    MPI_Gather(my_row.c_str(), max_len, MPI_BYTE, recv_buf, max_len, MPI_BYTE, 0, comm_);

    if (mpi_rank_ == 0) {
      std::ofstream ofs;
      OpenFileStream(ofs, fname, std::ios::out);
      csv_->DumpHeader(ofs);
      for (int r = 0; r < mpi_size; r++) {
        ofs << &recv_buf[r * len];
      }
      ofs.close();
    }
  }
};

#else

class RankCSV {
  std::unique_ptr<CSV> csv_;
  int mpi_rank_;
  
 public:
  RankCSV(std::initializer_list<std::string> cols)
  {
    std::vector<std::string> cols2 = cols;
    cols2.insert(cols2.begin(), "Rank");
    csv_.reset(new CSV(cols2, 1));

    this->At("Rank") = mpi_rank_;
  }
  
  Value &At(const std::string &col) {
    return csv_->At(col, 0);
  }

  void Dump(const std::string &fname) const {
    Dump(fname.c_str());
  }
  
  void Dump(std::ostream &os) const {
    csv_->Dump(os);
  }

  void Dump(const char *fname) const {
    std::ofstream ofs;
    OpenFileStream(ofs, fname, std::ios::out);
    Dump(ofs);
    ofs.close();
  }
};

#endif // USE_MPI

template<typename T>
std::string type_name()
{
  int status;
  std::string tname = typeid(T).name();
  char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
  if(status == 0) {
    tname = demangled_name;
    std::free(demangled_name);
  }
  return tname;
}

template<typename T>
std::string type_name(T& ) {
  return type_name<T>();
}

/**
 * \brief Utility class to extract function signature(return value and arity).
 */
template<class Signature>
struct function_traits;

template<typename R, typename...Args>
struct function_traits<R(Args...)> {
  using return_type = R;
  using arity = std::tuple<Args...>;
};

template<typename R, typename...Args>
struct function_traits<R(*)(Args...)> {
  using return_type = R;
  using arity = std::tuple<Args...>;
};

template<typename T, typename R, typename...Args>
struct function_traits<R(T::*)(Args...)> {
  using return_type = R;
  using arity = std::tuple<Args...>;
};

} // namespace util
} // namespace tapas

#endif // TAPAS_UTIL_H_
