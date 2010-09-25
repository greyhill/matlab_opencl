#include <ghp/util/cl.hpp>
#include <ghp/util/matlab.hpp>

#include <complex>
#include <iostream>
#include <list>

const char *source = " \
__kernel void mandelbrot(__constant float *input, \
    __global float *output, int num_iter) { \
  int id = get_global_id(0); \
  float c_real = input[2*id]; \
  float c_imag = input[2*id+1]; \
  float real = 0; \
  float imag = 0; \
  int iter; \
  for(iter=0; iter<num_iter; ++iter) { \
    if(real*real + imag*imag > 4) break; \
    real = real*real - imag*imag + c_real; \
    imag = 2*imag*real + c_imag; \
  } \
  output[id] = 1.0*iter/num_iter; \
} \
";

namespace matlab {

void mex_implementation(
    std::vector<matlab::mex_ref> &input,
    std::vector<matlab::mex_ref> &output) {
  if(input.size() < 2) {
    throw std::runtime_error("usage: mandelbrot(DATA, NUMITER)");
  }
  matlab::typed_array<double> input_data = input[0];
  double *input_real = input_data.real_ptr();
  double *input_imag = input_data.imag_ptr();
  std::vector<std::size_t> dims = input_data.get_dims();
  if(dims.size() != 2) {
    throw std::runtime_error("who do you think I am? I only do matrices!");
  }
  matlab::typed_array<double> num_iter_ref = input[1];
  int32_t num_iter = *matlab::typed_array<double>(input[1]).real_ptr();

  const std::size_t total_size = dims[0] * dims[1];
  std::vector<std::complex<float> > buffer;
  buffer.resize(total_size);
  for(std::size_t i=0; i<total_size; ++i) {
    buffer[i] = std::complex<float>(input_real[i], input_imag[i]);
  }

  std::list<cl::platform_ref> platforms;
  std::list<cl::device_ref> devices;
  cl::platform_ref::get_platforms(platforms);
  cl::platform_ref platform = platforms.front();
  platform.get_devices(devices);
  cl::device_ref device = devices.front();

  cl::context_ref context(platform, device);
  cl::command_queue_ref commands(context, device);
  cl::program_ref program(context, source);
  program.build();
  cl::kernel_ref kernel = program.get_kernel("mandelbrot");

  cl::buffer_ref input_buffer(context, 
      sizeof(std::complex<float>)*total_size,
      true, false, false);
  cl::buffer_ref output_buffer(context,
      sizeof(float)*total_size,
      false, true, false);

  cl::event_ref event = commands.write_buffer(input_buffer, 
      sizeof(std::complex<float>)*total_size,
      &buffer[0]);

  kernel.set_arg(0, input_buffer);
  kernel.set_arg(1, output_buffer);
  kernel.set_arg(2, num_iter);

  event.wait();
  std::size_t local_dims = 1;
  commands.run_kernel(kernel, 1, &total_size, &local_dims).wait();

  std::vector<float> buffer2;
  buffer2.resize(total_size);
  commands.read_buffer(output_buffer,
      sizeof(float)*total_size,
      &buffer2[0]).wait();
  matlab::typed_array<double> to_return(2, &dims[0], false);
  double *to_return_real = to_return.real_ptr();
  for(std::size_t i=0; i<total_size; ++i) {
    to_return_real[i] = buffer2[i];
  }

  if(output.size() < 1)
    throw std::runtime_error("i did the calculation but "
      "i guess you don't care ... :*-(");
  output[0] = to_return;
}

}

