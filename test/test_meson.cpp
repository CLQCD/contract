#include <runtime_api.h>
#include <contract.h>

void meson(void *correl, void *propag_a, void *propag_b, size_t volume, int gamma_ab, int gamma_dc)
{
  auto start = target_event_create();
  auto stop = target_event_create();
  meson_two_point(correl, propag_a, propag_b, volume, gamma_ab, gamma_dc);
  target_event_record(start);
  target_event_synchronize(start);
  for (int i = 0; i < 100; ++i) { meson_two_point(correl, propag_a, propag_b, volume, gamma_ab, gamma_dc); }
  target_event_record(stop);
  target_event_synchronize(stop);
  float milliseconds = target_event_elapsed_time(start, stop);
  printf("Time elapsed: %f ms, Bandwidth: %f GB/s\n", milliseconds / 100,
         (2.0 * volume * 4 * 4 * 3 * 3 * 16 + volume * 16) / (1024 * 1024 * 1024) / (milliseconds / 1000.0 / 100));
  target_event_destory(start);
  target_event_destory(stop);

  return;
}

int main(int argc, char *argv[])
{
  init(0);

  size_t volume = 24 * 24 * 24 * 18;
  void *correl = target_malloc(volume * 2 * sizeof(double));
  void *propag_i = target_malloc(volume * 16 * 9 * 2 * sizeof(double));
  void *propag_j = target_malloc(volume * 16 * 9 * 2 * sizeof(double));
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  target_free(correl);
  target_free(propag_i);
  target_free(propag_j);

  return 0;
}
