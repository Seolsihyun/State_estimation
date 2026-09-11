#include <IEKF.hpp>

#include <eigen3/Eigen/Dense>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using IEKF = iekf::IEKF;
using Matrix5d = IEKF::Matrix5d;
using Matrix15d = IEKF::Matrix15d;
using Timestamp = IEKF::Timestamp;
using Seconds = IEKF::Seconds;
using Vector3d = IEKF::Vector3d;

std::vector<double> parse_row(const std::string& line)
{
    std::vector<double> values;
    std::stringstream stream(line);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        values.push_back(std::stod(token));
    }
    return values;
}

void write_state(
    std::ofstream& output,
    double time,
    const Eigen::Matrix3d& rotation,
    const Vector3d& velocity,
    const Vector3d& position)
{
    output << std::setprecision(17) << time;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            output << ',' << rotation(row, col);
        }
    }
    for (int index = 0; index < 3; ++index)
    {
        output << ',' << velocity(index);
    }
    for (int index = 0; index < 3; ++index)
    {
        output << ',' << position(index);
    }
    output << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: ghaggin_cf231_runner INPUT.csv OUTPUT.csv\n";
        return 2;
    }

    std::ifstream input(argv[1]);
    if (!input)
    {
        throw std::runtime_error("cannot open input CSV");
    }
    std::ofstream output(argv[2]);
    if (!output)
    {
        throw std::runtime_error("cannot open output CSV");
    }

    std::string line;
    std::getline(input, line);  // header
    std::vector<std::vector<double>> rows;
    while (std::getline(input, line))
    {
        if (!line.empty())
        {
            auto values = parse_row(line);
            if (values.size() != 22)
            {
                throw std::runtime_error("each input row must have 22 columns");
            }
            rows.push_back(std::move(values));
        }
    }
    if (rows.size() < 2)
    {
        throw std::runtime_error("input must contain at least two IMU rows");
    }

    Matrix5d mean = Matrix5d::Identity();
    int column = 7;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            mean(row, col) = rows.front().at(column++);
        }
    }
    for (int index = 0; index < 3; ++index)
    {
        mean(index, 3) = rows.front().at(column++);
    }
    for (int index = 0; index < 3; ++index)
    {
        mean(index, 4) = rows.front().at(column++);
    }

    const double initial_time = rows.front().at(0);
    const Timestamp start{Seconds(initial_time)};
    IEKF filter(mean, Matrix15d::Identity(), start);

    output
        << "time,r00,r01,r02,r10,r11,r12,r20,r21,r22,"
        << "vx,vy,vz,px,py,pz\n";
    write_state(output, initial_time, filter.R(), filter.v(), filter.p());

    for (std::size_t index = 1; index < rows.size(); ++index)
    {
        const auto& values = rows[index];
        const Timestamp timestamp{Seconds(values[0])};
        const Vector3d acceleration(values[1], values[2], values[3]);
        const Vector3d gyroscope(values[4], values[5], values[6]);
        filter.addImu(timestamp, acceleration, gyroscope);
        write_state(
            output,
            values[0],
            filter.R(),
            filter.v(),
            filter.p());
    }
    return 0;
}
