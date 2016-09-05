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
#include <numeric>

#ifdef USE_MPI
# include <mpi.h>
#endif

namespace tapas {
namespace util {

template<class T>
T identity(T v) { return v; }

double stddev(std::vector<double> vals) {
  int n = (int)vals.size();
  double mean = std::accumulate(std::begin(vals), std::end(vals), 0) / n;
  double sigma = 0.0;

  for (double v : vals) {
    sigma += (mean - v) * (mean - v) / n;
  }

  return std::sqrt(sigma);
}


////////////////////////////////////////////////////////////////////////////////
//
// Check if a class has a certain member function
// 
////////////////////////////////////////////////////////////////////////////////

namespace {

template<typename T, bool B> struct Caller;

template<typename T> struct Caller<T, true> {
  static std::string Call(const T& t) { return t.label(); }
};

template<typename T> struct Caller<T, false> {
  static std::string Call(const T&) { return ""; }
};

} // namespace 

/**
 * \brief Returns a "label" of the given object. 
 * If class T has a method "std::string label() const", then calls it. Otherwise, returns "".
 */
template<class T>
struct GetLabel {
  template<typename U, std::string (U::*)() const> struct SFINAE {};
  template<typename U> static char Test(SFINAE<U, &U::label>*);
  template<typename U> static int Test(...);

  static constexpr bool HasFunc = sizeof(Test<T>(nullptr)) == sizeof(char);
  
  static std::string Value(const T& obj) {
    return Caller<T, HasFunc>::Call(obj);
  }
  static bool HasLabel() {
    return HasFunc;
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Performance reporting classes
// 
////////////////////////////////////////////////////////////////////////////////

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

class TimeRec {
  std::vector<std::string> cols_;
  std::vector<std::unordered_map<std::string, double>> table_;

  int GetMaxColWidth() const {
    int m = 0;
    for (auto &&c : cols_) {
      m = std::max(m, (int)c.size());
    }

    return std::max(12, m) + 2; // 15 is the width of scientific notations.
  }

  static std::string EscapeChars(const std::string &s_) {
    const std::vector<char> invalid_chars = {' ', '\t', '\r', '\n'};
    std::string s = s_;

    for (auto &&c: invalid_chars) {
      if (s.find(c) != std::string::npos) {
        std::cout << s.find(c) << std::endl;
        std::cerr << "Warning: TimeRec: column names cannot contain '"
                  << (c == ' ' ? "space" :
                      c == '\t' ? "\\t" :
                      c == '\r' ? "\\r" :
                      c == '\n' ? "\\n" : "")
                  << "'. They are replaced by '-'." << std::endl;
        
        while(s.find(c) != std::string::npos) {
          s = s.replace(s.find(c), 1, "-");
        }
      }
    }

    return s;
  }
  

  template<class T>
  static std::string Format(T &val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
  }

  static std::string Format(double val) {
    std::stringstream ss;
    ss << std::scientific << val;
    return ss.str();
  }
  
  static std::string Format(float val) {
    std::stringstream ss;
    ss << std::scientific << val;
    return ss.str();
  }


  template<class T>
  static void WriteValue(std::ostream &os, T val, int width) {
    std::string s = Format(val);
    int nsp = width - s.size();
    if (nsp >= 0) {
      for (int i = 0; i < nsp; i++) { os << " "; }
    }
    os << s;
  }
  
 public:
  TimeRec() : cols_(), table_() {}
  
  void Record(int time_step, std::string col, double val) {
    assert(time_step >= 0);

    if (std::find(cols_.begin(), cols_.end(), col) == cols_.end()) {
      cols_.push_back(col);
    }

    col = EscapeChars(col);

    if (table_.size() < (size_t)time_step + 1) { // assumes timestep starts from 0.
      table_.resize(time_step + 1);
    }

    table_[time_step][col] = val;
  }

  // Dump the CSV data to a file
  // all data is written by rank 0 process and the `os` file handler may be invalid
  // for other processes.
  void Dump(std::ostream &os, int mpi_rank, int mpi_size) const {
    if (cols_.size() == 0) return;
    
    int col_width = GetMaxColWidth();

    int max_ts = table_.size();

    std::stringstream ss;

    if (mpi_rank == 0) {
      WriteValue(ss, "TimeStep", col_width);
      WriteValue(ss, "Rank", col_width);
    
      for (const std::string &col : cols_) {
        WriteValue(ss, col, col_width);
      }
      ss << std::endl;
    }

    // Construct partial CSV data in each process
    for (int ts = 0; ts < max_ts; ts++) {
      const auto &row_map = table_[ts];

      WriteValue(ss, ts, col_width);
      WriteValue(ss, mpi_rank, col_width);

      for (const std::string &col : cols_) {
        double val = (row_map.find(col) == row_map.end())
                     ? 0
                     : row_map.at(col);
        
        WriteValue(ss, val, col_width);
      }
      ss << std::endl;
    }

    //// Gather the partial CSV data to rank 0

    // gather length and construct displ
    std::vector<int> recvcounts(mpi_size);
    std::vector<int> disps(mpi_size);

    int len = ss.str().size();
    MPI_Gather(&len, 1, MPI_INT, &(recvcounts[0]), 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < mpi_size; i++) {
      disps[i] = i == 0 ? 0 : recvcounts[i-1] + disps[i-1];
    }

    int total_length = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
    char *recv_buf = (mpi_rank == 0) ? new char[total_length + 1] : nullptr;

    // Gather CSV data
    char *send_buf = const_cast<char *>(ss.str().c_str());
    MPI_Gatherv(send_buf, ss.str().size(), MPI_BYTE,
                &(recv_buf[0]),  &(recvcounts[0]), &(disps[0]), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
      recv_buf[total_length] = '\0';
      os << recv_buf;
    }

    delete[] recv_buf;
  }
};

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


/**
 * \brief Sort multiple vectors together, with the first vector as keys.
 */
template<class T, class...Args>
void TiedSort(std::vector<T> &a, std::vector<Args>&... args) {
  using tt = struct {
    T key;
    std::tuple<Args...> val;
  };
  std::vector<tt> vals(a.size());

  for (size_t i = 0; i < a.size(); i++) {
    vals[i].key = a[i];
    vals[i].val = std::make_tuple(args[i]...);
  }

  std::sort(std::begin(vals), std::end(vals), [](const tt& a, const tt& b) {
      return a.key < b.key;
    });
  
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = vals[i].key;
    std::tuple<Args&...>(args[i]...) = vals[i].val;
  }
}



} // namespace util
} // namespace tapas

#endif // TAPAS_UTIL_H_
