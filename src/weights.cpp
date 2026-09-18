#include "weights.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

// --- Little-endian u64 ---
void writeU64(std::ostream &os, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    os.put((char)(v & 0xff));
    v >>= 8;
  }
}
uint64_t readU64(const std::string &buf, size_t &pos) {
  if (pos + 8 > buf.size()) throw std::runtime_error("weights: truncated u64");
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | (unsigned char)buf[pos + i];
  pos += 8;
  return v;
}

// --- JSON writer (strings + ints only) ---
std::string jsonEscape(const std::string &s) {
  std::string o;
  for (char c : s) {
    if (c == '"') o += "\\\"";
    else if (c == '\\') o += "\\\\";
    else if ((unsigned char)c < 0x20) {
      char tmp[8];
      snprintf(tmp, sizeof tmp, "\\u%04x", c);
      o += tmp;
    } else
      o += c;
  }
  return o;
}

// --- Minimal strict JSON parser for our schema ---
struct JsonParser {
  const std::string &s;
  size_t pos = 0;
  explicit JsonParser(const std::string &s_) : s(s_) {}
  void ws() {
    while (pos < s.size() && isspace((unsigned char)s[pos])) ++pos;
  }
  [[noreturn]] void fail(const std::string &w) {
    throw std::runtime_error("weights: bad JSON header (" + w + ")");
  }
  char peek() {
    ws();
    if (pos >= s.size()) fail("eof");
    return s[pos];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos;
  }
  std::string parseString() {
    expect('"');
    std::string o;
    while (pos < s.size()) {
      char c = s[pos++];
      if (c == '"') return o;
      if (c == '\\') {
        if (pos >= s.size()) fail("bad escape");
        char e = s[pos++];
        if (e == '"' || e == '\\' || e == '/') o += e;
        else if (e == 'n') o += '\n';
        else if (e == 't') o += '\t';
        else if (e == 'u') {
          if (pos + 4 > s.size()) fail("bad \\u");
          unsigned cp = 0;
          for (int i = 0; i < 4; ++i) {
            char h = s[pos++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= h - '0';
            else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
            else fail("bad \\u");
          }
          o += (char)cp;  // our files only escape ASCII controls
        } else
          fail("bad escape");
      } else
        o += c;
    }
    fail("unterminated string");
  }
  int64_t parseInt() {
    ws();
    size_t a = pos;
    if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
    if (pos >= s.size() || !isdigit((unsigned char)s[pos])) fail("bad int");
    while (pos < s.size() && isdigit((unsigned char)s[pos])) ++pos;
    return std::stoll(s.substr(a, pos - a));
  }
};

int64_t shapeProduct(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) throw std::runtime_error("weights: negative shape dim");
    n *= d;
  }
  return n;
}

}  // namespace

void saveWeights(const std::string &path,
                 const std::vector<WeightTensor> &tensors,
                 const std::map<std::string, std::string> &metadata,
                 bool fp32) {
  std::ostringstream header;
  header << "{\"__metadata__\":{";
  bool first = true;
  for (auto &[k, v] : metadata) {
    if (!first) header << ",";
    first = false;
    header << "\"" << jsonEscape(k) << "\":\"" << jsonEscape(v) << "\"";
  }
  header << "}";
  // data section: remember offsets after we know the header string
  struct Entry {
    std::string name, dtype;
    std::vector<int64_t> shape;
    uint64_t start, end;
  };
  std::vector<Entry> entries;
  uint64_t off = 0;
  const uint64_t elem = fp32 ? 4 : 8;
  for (auto &t : tensors) {
    uint64_t n = (uint64_t)shapeProduct(t.shape);
    if (fp32) {
      for (double v : t.data) {
        if (!std::isfinite(v) || std::fabs(v) > std::numeric_limits<float>::max())
          throw std::runtime_error("weights: F32 overflow/NaN in '" + t.name +
                                   "'");
      }
    }
    entries.push_back({t.name, fp32 ? "F32" : "F64", t.shape, off,
                       off + n * elem});
    off += n * elem;
  }
  for (auto &e : entries) {
    header << ",\"" << jsonEscape(e.name) << "\":{\"dtype\":\"" << e.dtype
           << "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) header << ",";
      header << e.shape[i];
    }
    header << "],\"data_offsets\":[" << e.start << "," << e.end << "]}";
  }
  header << "}";
  const std::string hs = header.str();

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("weights: cannot open " + path);
  writeU64(f, hs.size());
  f.write(hs.data(), hs.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (fp32) {
      for (double v : tensors[i].data) {
        float x = (float)v;
        f.write((const char *)&x, 4);
      }
    } else {
      f.write((const char *)tensors[i].data.data(),
              tensors[i].data.size() * 8);
    }
  }
  if (!f) throw std::runtime_error("weights: write failed " + path);
}

std::pair<std::vector<WeightTensor>, std::map<std::string, std::string>>
loadWeights(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("weights: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string buf = ss.str();
  size_t pos = 0;
  const uint64_t hlen = readU64(buf, pos);
  if (hlen > buf.size() - pos)
    throw std::runtime_error("weights: header length exceeds file");
  JsonParser p(buf);
  p.pos = pos;
  p.expect('{');
  std::map<std::string, std::string> metadata;
  struct Raw {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t start, end;
  };
  std::vector<std::pair<std::string, Raw>> raws;
  bool firstEntry = true;
  while (true) {
    if (p.peek() == '}') {
      ++p.pos;
      break;
    }
    if (!firstEntry) p.expect(',');
    firstEntry = false;
    std::string key = p.parseString();
    p.expect(':');
    if (key == "__metadata__") {
      p.expect('{');
      bool mfirst = true;
      while (true) {
        if (p.peek() == '}') {
          ++p.pos;
          break;
        }
        if (!mfirst) p.expect(',');
        mfirst = false;
        std::string mk = p.parseString();
        p.expect(':');
        metadata[mk] = p.parseString();
      }
    } else {
      Raw r;
      p.expect('{');
      bool ffirst = true;
      bool haveDtype = false, haveShape = false, haveOff = false;
      while (true) {
        if (p.peek() == '}') {
          ++p.pos;
          break;
        }
        if (!ffirst) p.expect(',');
        ffirst = false;
        std::string fk = p.parseString();
        p.expect(':');
        if (fk == "dtype") {
          r.dtype = p.parseString();
          haveDtype = true;
        } else if (fk == "shape") {
          p.expect('[');
          if (p.peek() != ']') {
            while (true) {
              r.shape.push_back(p.parseInt());
              if (p.peek() == ']') break;
              p.expect(',');
            }
          }
          p.expect(']');
          haveShape = true;
        } else if (fk == "data_offsets") {
          p.expect('[');
          r.start = (uint64_t)p.parseInt();
          p.expect(',');
          r.end = (uint64_t)p.parseInt();
          p.expect(']');
          haveOff = true;
        } else {
          p.fail("unexpected tensor field '" + fk + "'");
        }
      }
      if (!haveDtype || !haveShape || !haveOff)
        throw std::runtime_error("weights: incomplete entry '" + key + "'");
      if (r.dtype != "F32" && r.dtype != "F64")
        throw std::runtime_error("weights: bad dtype in '" + key + "'");
      raws.push_back({key, r});
    }
  }
  const size_t dataStart = pos + (size_t)hlen;
  std::vector<WeightTensor> out;
  for (auto &[name, r] : raws) {
    const uint64_t elem = r.dtype == "F32" ? 4 : 8;
    const int64_t n = shapeProduct(r.shape);
    if (r.end < r.start || r.end - r.start != (uint64_t)n * elem)
      throw std::runtime_error("weights: offset/size mismatch in '" + name +
                               "'");
    if (dataStart + r.end > buf.size())
      throw std::runtime_error("weights: truncated buffer in '" + name + "'");
    WeightTensor t;
    t.name = name;
    t.shape = r.shape;
    t.data.resize(n);
    const char *base = buf.data() + dataStart + r.start;
    if (r.dtype == "F64") {
      for (int64_t i = 0; i < n; ++i) {
        double v;
        __builtin_memcpy(&v, base + i * 8, 8);
        t.data[i] = v;
      }
    } else {
      for (int64_t i = 0; i < n; ++i) {
        float x;
        __builtin_memcpy(&x, base + i * 4, 4);
        t.data[i] = x;
      }
    }
    out.push_back(std::move(t));
  }
  return {out, metadata};
}
