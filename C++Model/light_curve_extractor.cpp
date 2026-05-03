#include <fitsio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct Image2D {
    long width = 0;
    long height = 0;
    std::vector<double> data;

    Image2D() = default;
    Image2D(long h, long w) : width(w), height(h), data(static_cast<size_t>(h * w), 0.0) {}

    double& operator()(long y, long x) {
        return data[static_cast<size_t>(y * width + x)];
    }

    double operator()(long y, long x) const {
        return data[static_cast<size_t>(y * width + x)];
    }

    bool empty() const {
        return data.empty() || width <= 0 || height <= 0;
    }
};


struct SigmaStats {
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
};

struct FitsMeta {
    std::string date_obs;
    std::optional<double> ra_deg;
    std::optional<double> dec_deg;

    bool has_wcs = false;
    double crpix1 = 0.0;
    double crpix2 = 0.0;
    double crval1 = 0.0;
    double crval2 = 0.0;
    double cd11 = 0.0;
    double cd12 = 0.0;
    double cd21 = 0.0;
    double cd22 = 0.0;
};

class LightCurveExtractor {
public:
    LightCurveExtractor(const Image2D& bias_,
                        const Image2D& dark_,
                        const Image2D& flat_,
                        const std::string& data_map_,
                        const std::string& star_name_,
                        const std::optional<std::pair<double, double>>& centroid_override_ = std::nullopt)
        : bias(bias_),
          dark(dark_),
          flat(flat_),
          data_map(data_map_),
          star_name(star_name_),
          Centroid_override(centroid_override_) {
        file_list = load_csv_file_list(data_map);
    }

    Image2D load_calibrated(const Image2D& input) const {
        Image2D out(input.height, input.width);

        double flat_med = median(flat.data);
        if (flat_med == 0.0) {
            flat_med = 1.0;
        }

        for (long y = 0; y < input.height; ++y) {
            for (long x = 0; x < input.width; ++x) {
                double flat_norm = flat(y, x) / flat_med;
                if (std::abs(flat_norm) < 1e-12) {
                    flat_norm = 1.0;
                }

                try {
                    out(y, x) = (input(y, x) - bias(y, x) - dark(y, x)) / flat_norm;
                } catch (...) {
                    out(y, x) = input(y, x) - dark(y, x);
                }
            }
        }

        std::cout << "Calibrated image created.\n";
        return out;
    }

    std::pair<long, long> get_max(const Image2D& img) const {
        return {img.height / 2, img.width / 2};
    }

    std::pair<long, long> get_midpoint(const Image2D& img) const {
        return {img.width / 2, img.height / 2};
    }

    std::pair<std::vector<double>, std::vector<double>>
    drop_outliers(const std::vector<double>& data_input,
                  const std::vector<double>& time_input,
                  double threshold = 2.0) const {
        if (data_input.size() != time_input.size() || data_input.empty()) {
            return {data_input, time_input};
        }

        double m = mean(data_input);
        double s = stddev(data_input, m);

        if (s < 1e-12) {
            return {data_input, time_input};
        }

        std::vector<double> filtered_data;
        std::vector<double> filtered_time;

        for (size_t i = 0; i < data_input.size(); ++i) {
            double z = std::abs((data_input[i] - m) / s);
            if (z <= threshold) {
                filtered_data.push_back(data_input[i]);
                filtered_time.push_back(time_input[i]);
            }
        }
        std::cout << "Dropped " << (data_input.size() - filtered_data.size()) << " outliers.\n";
        return {filtered_data, filtered_time};
    }

    std::pair<std::vector<double>, std::vector<double>> extract_light_curve(bool normalize = false) {
        std::optional<std::pair<double, double>> cords;
        std::optional<double> ra;
        std::optional<double> dec;

        for (const auto& file_path : file_list) {
            std::cout << file_path << "\n";

            FitsMeta meta;
            Image2D raw = read_fits(file_path, meta);
            if (raw.empty()) {
                std::cerr << "Failed to read FITS: " << file_path << "\n";
                continue;
            }

            Image2D calibrated_second = raw;

            if (reference_frame.empty()) {
                reference_frame = calibrated_second;
            }

            std::string date_obs = meta.date_obs;
            if (date_obs.empty()) {
                std::cerr << "Missing DATE-AVG/DATE-OBS in " << file_path << "\n";
                date_obs = "2000-01-01T00:00:00";
            }

            std::string date_obs_clean = cleanup_date(date_obs);
            double jd_utc = isot_to_julian(date_obs_clean);

            // Approximation of Python's TDB-based BTJD.
            // Exact UTC->TDB needs ERFA/SOFA or astropy-equivalent routines.
            double btjd = jd_utc - 2457000.0;
            date_array.push_back(btjd);

            ra = meta.ra_deg;
            dec = meta.dec_deg;

            try {
                if (meta.has_wcs && ra.has_value() && dec.has_value()) {
                    auto pxpy = world_to_pixel_linear(meta, ra.value(), dec.value());
                    cords = {pxpy.second, pxpy.first}; // store as (row, col)-like shape
                    std::cout << "WCS coords: (" << cords->first << ", " << cords->second << ")\n";
                }
            } catch (const std::exception& e) {
                std::cout << "WCS conversion failed: " << e.what() << "\n";
                std::cout << "Using last known coordinates or image center.\n";
            }

            if (Centroid_override.has_value()) {
                auto max_idx = argmax(calibrated_second);
                cords = {static_cast<double>(max_idx.first), static_cast<double>(max_idx.second)};
                std::cout << "Using Centroid Override: (" << cords->first << ", " << cords->second << ")\n";
                allignment_needed += 1;
            }

            if (!cords.has_value()) {
                auto mid = get_midpoint(calibrated_second);
                cords = {static_cast<double>(mid.second), static_cast<double>(mid.first)};
            }

            auto cords_transformed = cords.value();

            long ny = calibrated_second.height;
            long nx = calibrated_second.width;
            std::cout << nx << " " << ny << "\n";

            int x = static_cast<int>(cords_transformed.first);
            int y = static_cast<int>(cords_transformed.second);

            int half_box = 50;
            int x1 = std::max(0, x - half_box);
            int x2 = std::min(static_cast<int>(nx), x + half_box);
            int y1 = std::max(0, y - half_box);
            int y2 = std::min(static_cast<int>(ny), y + half_box);

            std::cout << y1 << " " << y2 << " " << x1 << " " << x2 << "\n";

            auto past_cord = cords_transformed;

            if (!cords.has_value()) {
                cords_transformed = past_cord;
            } else {
                double background = *std::min_element(calibrated_second.data.begin(), calibrated_second.data.end());
                (void)background;

                Image2D data_background_subtracted = crop(calibrated_second, x1, x2, y1, y2);

                SigmaStats stats = sigma_clipped_stats(data_background_subtracted.data, 3.0, 5);
                for (double& v : data_background_subtracted.data) {
                    v -= stats.median;
                }

                auto centroid_local = centroid_1dg(data_background_subtracted);
                double x4 = centroid_local.first;
                double y4 = centroid_local.second;

                cords_transformed = {x4 + y1, y4 + x1};

                std::vector<int> radii;
                for (int r = 1; r < 15; ++r) {
                    radii.push_back(r);
                }

                std::vector<double> profile;
                for (int r : radii) {
                    profile.push_back(circular_sum(data_background_subtracted, x4, y4, r));
                }

                std::vector<double> growth_rate;
                for (size_t i = 1; i < profile.size(); ++i) {
                    growth_rate.push_back(profile[i] - profile[i - 1]);
                }

                std::vector<double> second_derivative;
                for (size_t i = 1; i < growth_rate.size(); ++i) {
                    second_derivative.push_back(growth_rate[i] - growth_rate[i - 1]);
                }

                int optimal_index = -1;
                for (size_t i = 0; i < second_derivative.size(); ++i) {
                    if (second_derivative[i] < 0.0) {
                        optimal_index = static_cast<int>(i);
                        break;
                    }
                }

                if (optimal_index < 0) {
                    optimal_index = static_cast<int>(second_derivative.size() / 2);
                }

                if (optimal_index >= static_cast<int>(radii.size())) {
                    optimal_index = static_cast<int>(radii.size()) - 1;
                }

                int optimal_radius = radii[std::max(0, optimal_index)];
                double ap_radius = static_cast<double>(optimal_radius);
                double ann_inner = optimal_radius + 5.0;
                double ann_width = 3.0;
                double ann_outer = ann_inner + ann_width;

                double aperture_sum_0 = circular_sum(data_background_subtracted, x4, y4, ap_radius);
                double aperture_sum_1 = annulus_sum(data_background_subtracted, x4, y4, ann_inner, ann_outer);

                double area = M_PI * ann_outer * ann_outer - M_PI * ann_inner * ann_inner;
                double ap_area = M_PI * ap_radius * ap_radius;

                double bkg_mean = (area > 0.0) ? (aperture_sum_1 / area) : 0.0;
                double bkg_sum = bkg_mean * ap_area;
                double final_sum = aperture_sum_0 - bkg_sum;

                if (ap_area > 0.0) {
                    final_sum /= ap_area;
                }

                saving_constant += 1;
                photom_list.push_back(final_sum);

                std::cout << "Flux list size: " << photom_list.size() << "\n";
            }
        }

        auto filtered = drop_outliers(photom_list, date_array, 2.0);
        photom_list = filtered.first;
        date_array = filtered.second;

        if (normalize) {
            photom_list = normalize_lightcurve();
        }

        std::cout << "Final light curve points: " << photom_list.size() << "\n";
        return {photom_list, date_array};
    }

    void save_lightcurve(const std::string& output_path = "") const {
        if (photom_list.empty() || date_array.empty()) {
            std::cout << "No light curve data to save. Please run extract_light_curve() first.\n";
            return;
        }

        if (output_path.empty()) {
            std::cout << "No output path provided\n";
            return;
        }

        std::ofstream out(output_path);
        out << "Time,Flux\n";
        for (size_t i = 0; i < photom_list.size(); ++i) {
            out << std::setprecision(15) << date_array[i] << "," << photom_list[i] << "\n";
        }
        std::cout << "Light curve saved to " << output_path << "\n";
    }

    void plot_lightcurve() const {
        if (photom_list.empty() || date_array.empty()) {
            std::cout << "No light curve data to plot. Please run extract_light_curve() first.\n";
            return;
        }

        std::cout << "Plotting light curve is not implemented in this pure C++ version.\n";
        std::cout << "Use save_lightcurve(...) and plot externally with Python, gnuplot, or ROOT.\n";
    }

    std::vector<std::pair<double, double>> tuple_format() const {
        if (photom_list.empty() || date_array.empty()) {
            std::cout << "No light curve data to format. Please run extract_light_curve() first.\n";
            return {};
        }

        std::vector<std::pair<double, double>> out;
        for (size_t i = 0; i < photom_list.size(); ++i) {
            out.emplace_back(date_array[i], photom_list[i]);
        }
        return out;
    }

    std::vector<double> normalize_lightcurve() const {
        if (photom_list.empty()) {
            std::cout << "No light curve data to normalize. Please run extract_light_curve() first.\n";
            return {};
        }

        double min_v = *std::min_element(photom_list.begin(), photom_list.end());
        double max_v = *std::max_element(photom_list.begin(), photom_list.end());

        if (std::abs(max_v - min_v) < 1e-12) {
            return photom_list;
        }

        std::vector<double> normalized;
        normalized.reserve(photom_list.size());

        for (double v : photom_list) {
            normalized.push_back(2.0 * (v - min_v) / (max_v - min_v) - 1.0);
        }

        return normalized;
    }

    const std::vector<double>& get_photom_list() const { return photom_list; }
    const std::vector<double>& get_date_array() const { return date_array; }

private:
    Image2D bias;
    Image2D dark;
    Image2D flat;
    std::string data_map;
    std::string star_name;

    std::vector<std::string> file_list;
    Image2D reference_frame;

    int saving_constant = 0;
    int allignment_needed = 0;
    std::optional<std::pair<double, double>> Centroid_override;

    std::vector<double> photom_list;
    std::vector<double> date_array;

private:
    static std::vector<std::string> load_csv_file_list(const std::string& csv_path) {
        std::ifstream file(csv_path);
        if (!file) {
            throw std::runtime_error("Failed to open CSV: " + csv_path);
        }

        std::string header;
        std::getline(file, header);

        std::vector<std::string> columns = split_csv_line(header);
        int fits_col = -1;
        for (size_t i = 0; i < columns.size(); ++i) {
            if (columns[i] == "FITS File Path") {
                fits_col = static_cast<int>(i);
                break;
            }
        }

        if (fits_col < 0) {
            throw std::runtime_error("CSV does not contain 'FITS File Path' column.");
        }

        std::vector<std::string> paths;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            std::vector<std::string> row = split_csv_line(line);
            if (fits_col < static_cast<int>(row.size())) {
                paths.push_back(row[fits_col]);
            }
        }

        return paths;
    }

    static std::vector<std::string> split_csv_line(const std::string& line) {
        std::vector<std::string> out;
        std::string cur;
        bool in_quotes = false;

        for (char ch : line) {
            if (ch == '"') {
                in_quotes = !in_quotes;
            } else if (ch == ',' && !in_quotes) {
                out.push_back(trim(cur));
                cur.clear();
            } else {
                cur.push_back(ch);
            }
        }
        out.push_back(trim(cur));

        for (std::string& s : out) {
            if (!s.empty() && s.front() == '"' && s.back() == '"' && s.size() >= 2) {
                s = s.substr(1, s.size() - 2);
            }
        }

        return out;
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end = s.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) {
            return "";
        }
        return s.substr(start, end - start + 1);
    }

    static Image2D read_fits(const std::string& path, FitsMeta& meta) {
        fitsfile* fptr = nullptr;
        int status = 0;
        int naxis = 0;
        long naxes[2] = {0, 0};

        fits_open_file(&fptr, path.c_str(), READONLY, &status);
        if (status) {
            fits_report_error(stderr, status);
            return {};
        }

        read_header_string(fptr, "DATE-AVG", meta.date_obs);
        if (meta.date_obs.empty()) {
            read_header_string(fptr, "DATE-OBS", meta.date_obs);
        }

        double ra_tmp = 0.0;
        double dec_tmp = 0.0;

        if (read_header_double(fptr, "OBJCTRA", ra_tmp)) {
            meta.ra_deg = ra_tmp;
        }
        if (read_header_double(fptr, "OBJCTDEC", dec_tmp)) {
            meta.dec_deg = dec_tmp;
        }

        if (!meta.ra_deg.has_value() && read_header_double(fptr, "CRVAL1", ra_tmp)) {
            meta.ra_deg = ra_tmp;
        }
        if (!meta.dec_deg.has_value() && read_header_double(fptr, "CRVAL2", dec_tmp)) {
            meta.dec_deg = dec_tmp;
        }

        double crpix1 = 0.0, crpix2 = 0.0, crval1 = 0.0, crval2 = 0.0;
        bool ok_crpix1 = read_header_double(fptr, "CRPIX1", crpix1);
        bool ok_crpix2 = read_header_double(fptr, "CRPIX2", crpix2);
        bool ok_crval1 = read_header_double(fptr, "CRVAL1", crval1);
        bool ok_crval2 = read_header_double(fptr, "CRVAL2", crval2);

        double cd11 = 0.0, cd12 = 0.0, cd21 = 0.0, cd22 = 0.0;
        bool has_cd11 = read_header_double(fptr, "CD1_1", cd11);
        bool has_cd12 = read_header_double(fptr, "CD1_2", cd12);
        bool has_cd21 = read_header_double(fptr, "CD2_1", cd21);
        bool has_cd22 = read_header_double(fptr, "CD2_2", cd22);

        if (!(has_cd11 && has_cd12 && has_cd21 && has_cd22)) {
            double cdelt1 = 0.0, cdelt2 = 0.0;
            bool has_cdelt1 = read_header_double(fptr, "CDELT1", cdelt1);
            bool has_cdelt2 = read_header_double(fptr, "CDELT2", cdelt2);

            if (has_cdelt1 && has_cdelt2) {
                cd11 = cdelt1;
                cd12 = 0.0;
                cd21 = 0.0;
                cd22 = cdelt2;
                has_cd11 = has_cd12 = has_cd21 = has_cd22 = true;
            }
        }

        if (ok_crpix1 && ok_crpix2 && ok_crval1 && ok_crval2 && has_cd11 && has_cd12 && has_cd21 && has_cd22) {
            meta.has_wcs = true;
            meta.crpix1 = crpix1;
            meta.crpix2 = crpix2;
            meta.crval1 = crval1;
            meta.crval2 = crval2;
            meta.cd11 = cd11;
            meta.cd12 = cd12;
            meta.cd21 = cd21;
            meta.cd22 = cd22;
        }

        fits_get_img_dim(fptr, &naxis, &status);
        fits_get_img_size(fptr, 2, naxes, &status);
        if (status || naxis != 2) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return {};
        }

        long nelements = naxes[0] * naxes[1];
        std::vector<double> buffer(static_cast<size_t>(nelements));
        long fpixel[2] = {1, 1};

        int anynul = 0;
        fits_read_pix(fptr, TDOUBLE, fpixel, nelements, nullptr, buffer.data(), &anynul, &status);
        fits_close_file(fptr, &status);

        if (status) {
            fits_report_error(stderr, status);
            return {};
        }

        Image2D img(naxes[1], naxes[0]);
        img.data = std::move(buffer);
        return img;
    }

    static bool read_header_string(fitsfile* fptr, const char* key, std::string& out) {
        int status = 0;
        char value[FLEN_VALUE] = {0};
        if (fits_read_key(fptr, TSTRING, const_cast<char*>(key), value, nullptr, &status) == 0) {
            out = trim(value);
            return true;
        }
        return false;
    }

    static bool read_header_double(fitsfile* fptr, const char* key, double& out) {
        int status = 0;
        double value = 0.0;
        if (fits_read_key(fptr, TDOUBLE, const_cast<char*>(key), &value, nullptr, &status) == 0) {
            out = value;
            return true;
        }
        return false;
    }

    static std::string cleanup_date(std::string date_obs) {
        auto pos = date_obs.find("'/");
        if (pos != std::string::npos) {
            date_obs = date_obs.substr(0, pos);
        }

        date_obs = trim(date_obs);

        auto dot = date_obs.find('.');
        if (dot != std::string::npos) {
            std::string left = date_obs.substr(0, dot);
            std::string right = date_obs.substr(dot + 1);
            if (right.size() > 6) {
                right = right.substr(0, 6);
            }
            date_obs = left + "." + right;
        }

        return date_obs;
    }

    static double isot_to_julian(const std::string& isot) {
        int year = 0, month = 0, day = 0, hour = 0, minute = 0;
        double second = 0.0;
        char sep1 = '-', sep2 = '-', sep3 = 'T', sep4 = ':', sep5 = ':';

        std::stringstream ss(isot);
        ss >> year >> sep1 >> month >> sep2 >> day >> sep3 >> hour >> sep4 >> minute >> sep5 >> second;

        if (!ss) {
            throw std::runtime_error("Failed to parse ISO timestamp: " + isot);
        }

        if (month <= 2) {
            year -= 1;
            month += 12;
        }

        int A = year / 100;
        int B = 2 - A + (A / 4);

        double day_fraction = (hour + minute / 60.0 + second / 3600.0) / 24.0;

        double jd = std::floor(365.25 * (year + 4716))
                  + std::floor(30.6001 * (month + 1))
                  + day + day_fraction + B - 1524.5;

        return jd;
    }

    static std::pair<double, double> world_to_pixel_linear(const FitsMeta& meta, double ra_deg, double dec_deg) {
        double dra = ra_deg - meta.crval1;
        double ddec = dec_deg - meta.crval2;

        double det = meta.cd11 * meta.cd22 - meta.cd12 * meta.cd21;
        if (std::abs(det) < 1e-20) {
            throw std::runtime_error("Singular WCS CD matrix");
        }

        double inv11 =  meta.cd22 / det;
        double inv12 = -meta.cd12 / det;
        double inv21 = -meta.cd21 / det;
        double inv22 =  meta.cd11 / det;

        double dx = inv11 * dra + inv12 * ddec;
        double dy = inv21 * dra + inv22 * ddec;

        double xpix = meta.crpix1 + dx - 1.0;
        double ypix = meta.crpix2 + dy - 1.0;

        return {xpix, ypix};
    }

    static std::pair<long, long> argmax(const Image2D& img) {
        long best_y = 0;
        long best_x = 0;
        double best_v = -std::numeric_limits<double>::infinity();

        for (long y = 0; y < img.height; ++y) {
            for (long x = 0; x < img.width; ++x) {
                double v = img(y, x);
                if (v > best_v) {
                    best_v = v;
                    best_y = y;
                    best_x = x;
                }
            }
        }

        return {best_y, best_x};
    }

    static Image2D crop(const Image2D& img, int x1, int x2, int y1, int y2) {
        int w = std::max(0, x2 - x1);
        int h = std::max(0, y2 - y1);

        Image2D out(h, w);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                out(y, x) = img(y1 + y, x1 + x);
            }
        }
        return out;
    }

    static SigmaStats sigma_clipped_stats(const std::vector<double>& input, double sigma = 3.0, int iterations = 5) {
        if (input.empty()) {
            return {};
        }

        std::vector<double> work = input;

        for (int iter = 0; iter < iterations; ++iter) {
            double m = mean(work);
            double s = stddev(work, m);
            if (s < 1e-12) {
                break;
            }

            std::vector<double> clipped;
            clipped.reserve(work.size());

            for (double v : work) {
                if (std::abs(v - m) <= sigma * s) {
                    clipped.push_back(v);
                }
            }

            if (clipped.size() == work.size() || clipped.empty()) {
                break;
            }

            work = std::move(clipped);
        }

        SigmaStats out;
        out.mean = mean(work);
        out.median = median(work);
        out.stddev = stddev(work, out.mean);
        return out;
    }

    static std::pair<double, double> centroid_1dg(const Image2D& img) {
        std::vector<double> proj_x(static_cast<size_t>(img.width), 0.0);
        std::vector<double> proj_y(static_cast<size_t>(img.height), 0.0);

        for (long y = 0; y < img.height; ++y) {
            for (long x = 0; x < img.width; ++x) {
                double v = img(y, x);
                if (v < 0.0) {
                    v = 0.0;
                }
                proj_x[static_cast<size_t>(x)] += v;
                proj_y[static_cast<size_t>(y)] += v;
            }
        }

        double sum_x = std::accumulate(proj_x.begin(), proj_x.end(), 0.0);
        double sum_y = std::accumulate(proj_y.begin(), proj_y.end(), 0.0);

        if (sum_x < 1e-12 || sum_y < 1e-12) {
            return {img.width / 2.0, img.height / 2.0};
        }

        double cx = 0.0;
        double cy = 0.0;

        for (size_t x = 0; x < proj_x.size(); ++x) {
            cx += static_cast<double>(x) * proj_x[x];
        }
        for (size_t y = 0; y < proj_y.size(); ++y) {
            cy += static_cast<double>(y) * proj_y[y];
        }

        cx /= sum_x;
        cy /= sum_y;

        return {cx, cy};
    }

    static double circular_sum(const Image2D& img, double cx, double cy, double radius) {
        double r2 = radius * radius;
        double sum = 0.0;

        for (long y = 0; y < img.height; ++y) {
            for (long x = 0; x < img.width; ++x) {
                double dx = static_cast<double>(x) - cx;
                double dy = static_cast<double>(y) - cy;
                if (dx * dx + dy * dy <= r2) {
                    sum += img(y, x);
                }
            }
        }

        return sum;
    }

    static double annulus_sum(const Image2D& img, double cx, double cy, double r_in, double r_out) {
        double r_in2 = r_in * r_in;
        double r_out2 = r_out * r_out;
        double sum = 0.0;

        for (long y = 0; y < img.height; ++y) {
            for (long x = 0; x < img.width; ++x) {
                double dx = static_cast<double>(x) - cx;
                double dy = static_cast<double>(y) - cy;
                double rr = dx * dx + dy * dy;
                if (rr >= r_in2 && rr <= r_out2) {
                    sum += img(y, x);
                }
            }
        }

        return sum;
    }

    static double mean(const std::vector<double>& v) {
        if (v.empty()) {
            return 0.0;
        }
        return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    }

    static double stddev(const std::vector<double>& v, double m) {
        if (v.size() < 2) {
            return 0.0;
        }

        double acc = 0.0;
        for (double x : v) {
            double d = x - m;
            acc += d * d;
        }
        return std::sqrt(acc / static_cast<double>(v.size()));
    }

    static double median(std::vector<double> v) {
        if (v.empty()) {
            return 0.0;
        }

        size_t n = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + n, v.end());
        double med = v[n];

        if (v.size() % 2 == 0) {
            auto max_it = std::max_element(v.begin(), v.begin() + n);
            med = 0.5 * (med + *max_it);
        }

        return med;
    }
};

