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
#include <set>
#include <unordered_set>

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

bool FileExists(const std::string &name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

std::string IncrPostfix(const std::string &fname) {
  // find postfix like ".1" ".21"
  if (fname.size() == 0) {
    return fname;
  }

  int tail = fname.size() - 1;
  std::string num_part = "";

  while (tail >= 0) {
    if (isdigit(fname[tail])) {
      num_part.insert(num_part.begin(), fname[tail]);
      tail--;
    } else if (fname[tail] == '.') {
      break;
    } else {
      return fname + ".0";
    }
  }

  int num = atoi(num_part.c_str()) + 1;
  std::stringstream ss;
  ss << std::string(&(fname[0]), &(fname[tail])) << "." << num;
  return ss.str();
}

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

  enum class FileFormat {
    CSV,
    FWF,
  };

  //! \brief Write a value to a stream either in CSV or FWF format.
  template<class T>
  static void WriteValue(std::ostream &os, T val, FileFormat fmt, char sep, bool last_col,  int width) {
    std::string s = Format(val);

    if (fmt == FileFormat::FWF) {
      int nsp = width - s.size();
      if (nsp >= 0) {
        for (int i = 0; i < nsp; i++) { os << " "; }
      }
    }
    
    os << s; // value

    if (fmt == FileFormat::CSV && !last_col) {
      os << sep;
    }
  }
  
  /**
   * \brief Read envvar 'TAPAS_REPORT_FORMAT' and returns either FileFormat::CSV or FileFormat::FWF.
   *
   * \param [out] fmt 
   * \param [out] sep Separator character if FileFormat::CSV. The default value is ','
   */
  static void GetFileFormat(FileFormat &fmt, char &sep) {
    const char *env_fmt = getenv("TAPAS_REPORT_FORMAT");

    if (env_fmt != nullptr) {
      std::string s = env_fmt;
      std::transform(s.begin(), s.end(), s.begin(), ::tolower); // convert to lower case
      
      if (s == "fwf") {
        fmt = FileFormat::FWF;
        return;
      }

      // default value is "CSV"
      if (s != "csv") {
        std::cerr << "Warning: Unknown reporting file format from TAPAS_REPORT_FORMAT: '" << s << "'" << std::endl;
      }

      fmt = FileFormat::CSV;

      const char *env_sep = getenv("TAPAS_REPORT_SEP");
      if (env_sep != nullptr && strlen(env_sep) >= 1) {
        sep = env_sep[0];

        if (strlen(env_sep) > 1) {
          std::cerr << "Warning: Separator must be a single character" << std::endl;
        }
      } else {
        sep = ',';
      }
    } else {
      fmt = FileFormat::CSV;
      sep = ',';
      return;
    }
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

  //! \brief Dump the data to a file. File format is CSV or FWF depending on TAPAS_REPORT_FILE_FORMAT
  //
  // All data is written by rank 0 process and the `os` file handler is invalid
  // for other processes.
  void Dump(std::ostream &os, int mpi_rank, int mpi_size) const {
    if (cols_.size() == 0) return;
    
    int col_width = GetMaxColWidth();

    int max_ts = table_.size();

    std::stringstream ss;
    
    FileFormat fmt;
    char sep;
    GetFileFormat(fmt, sep);

    if (mpi_rank == 0) {
      WriteValue(ss, "TimeStep", fmt, sep, false, col_width);
      WriteValue(ss, "Rank", fmt, sep, false, col_width);

      for (size_t i = 0; i < cols_.size(); i++) {
        WriteValue(ss, cols_[i], fmt, sep, i == cols_.size()-1, col_width);
      }
      ss << std::endl;
    }

    // Construct partial CSV data in each process
    for (int ts = 0; ts < max_ts; ts++) {
      const auto &row_map = table_[ts];

      WriteValue(ss, ts, fmt, sep, false, col_width);
      WriteValue(ss, mpi_rank, fmt, sep, false, col_width);

      for (size_t i = 0; i < cols_.size(); i++) {
        auto col = cols_[i];
        double val = (row_map.find(col) == row_map.end())
                     ? 0
                     : row_map.at(col);
        
        WriteValue(ss, val, fmt, sep, i == cols_.size()-1, col_width);
      }
      ss << std::endl;
    }

    //// Gather the partial CSV data to rank 0

    // gather lengths of lines and construct displ
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

/**
 * \brief Sort multiple vectors together, with the first vector as keys. (downgraded version)
 * For some non-C++11-ready compileres (including nvcc)
 */
template<class T1, class T2>
void TiedSort2(std::vector<T1> &keys, std::vector<T2>& a2) {
  TAPAS_ASSERT(keys.size() == a2.size());
  
  using tt = struct {
    T1 key;
    T2 val;
  };
  std::vector<tt> vals(a2.size());

  for (size_t i = 0; i < a2.size(); i++) {
    vals[i].key = keys[i];
    vals[i].val = a2[i];
  }

  std::sort(std::begin(vals), std::end(vals), [](const tt& a, const tt& b) {
      return a.key < b.key;
    });
  
  for (size_t i = 0; i < a2.size(); i++) {
    keys[i] = vals[i].key;
    a2[i] = vals[i].val;
  }
}

template<class FpType, int BYTES>
struct NearZeroFP;

template<class FpType>
struct NearZeroFP<FpType, 4> {
  static const constexpr int Value = -24;
};

template<class FpType>
struct NearZeroFP<FpType, 8> {
  static const constexpr int Value = -52;
};

template<class FpType>
struct NearZeroFP<FpType, 16> {
  static const constexpr int Value = -52;
};

template<class FP>
FP NearZeroValue(FP a) {
  int deg = NearZeroFP<FP, sizeof(FP)>::Value;
  while(1) {
    FP v = a * pow(2, deg);
    //printf("deg=%d, v = %.16f, a+v = %.16f, a < a+v => %d\n", deg, v, a+v, a<a+v);
    if (a == a + v) {
      deg++;
      continue;
    } else {
      return v;
    }
  }
}

/**
 * \brief Compute a difference of two sets (include std::unordered_set)
 */
template<class T>
std::set<T> SetDiff(const std::set<T> &a, const std::set<T> &b) {
  std::set<T> s;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(s, s.end()));

  return s;
}

template<class T>
std::unordered_set<T> SetDiff(const std::unordered_set<T> &a, const std::unordered_set<T> &b) {
  std::unordered_set<T> s;
  std::copy_if(a.begin(), a.end(), std::inserter(s, s.end()),
               [&b](const T &e) { return b.count(e) == 0; });

  return s;
}

template<class MapType, class ReturnType>
ReturnType GetKeys(const MapType &map) {
  using KT = typename MapType::key_type;
  using VT = typename MapType::mapped_type;
  ReturnType v;
  v.clear();
  v.reserve(map.size());
  std::transform(map.begin(), map.end(), std::inserter(v, v.end()),
                 [](std::pair<KT, VT> kv) { return kv.first; });
  return v;
}

template <typename Func, typename... Args>
struct is_callable
{
private:
  template <typename CheckType> static std::true_type check(decltype(std::declval<CheckType>()(std::declval<Args>()...)) *);
  template <typename CheckType> static std::false_type check(...);
 public:
  using type = decltype(check<Func>(nullptr));
  static constexpr bool value = type::value;
};

} // namespace util
} // namespace tapas

#endif // TAPAS_UTIL_H_
