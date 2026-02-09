#include <runtime_api.h>
#include <contract.h>

void proton_v2(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
               size_t volume, int gamma_ij, int gamma_kl, int gamma_mn)
{
  auto start = target_event_create();
  auto stop = target_event_create();
  baryon_two_point_v2(correl, propag_i, propag_j, propag_m, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
  target_event_record(start);
  target_event_synchronize(start);
  for (int i = 0; i < 100; ++i) {
    baryon_two_point_v2(correl, propag_i, propag_j, propag_m, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
  }
  target_event_record(stop);
  target_event_synchronize(stop);
  float milliseconds = target_event_elapsed_time(start, stop);
  printf("Time elapsed: %f ms, Bandwidth: %f GB/s\n", milliseconds / 100,
         (3.0 * volume * 4 * 4 * 3 * 3 * 16 + volume * 16) / (1024 * 1024 * 1024) / (milliseconds / 1000.0 / 100));
  target_event_destory(start);
  target_event_destory(stop);

  return;
}

void proton(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
            size_t volume, int gamma_ij, int gamma_kl, int gamma_mn)
{
  auto start = target_event_create();
  auto stop = target_event_create();
  baryon_two_point(correl, propag_i, propag_j, propag_m, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
  target_event_record(start);
  target_event_synchronize(start);
  for (int i = 0; i < 100; ++i) {
    baryon_two_point(correl, propag_i, propag_j, propag_m, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
  }
  target_event_record(stop);
  target_event_synchronize(stop);
  float milliseconds = target_event_elapsed_time(start, stop);
  printf("Time elapsed: %f ms, Bandwidth: %f GB/s\n", milliseconds / 100,
         (3.0 * volume * 4 * 4 * 3 * 3 * 16 + volume * 16) / (1024 * 1024 * 1024) / (milliseconds / 1000.0 / 100));
  target_event_destory(start);
  target_event_destory(stop);

  return;
}

void proton_general(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                    size_t volume, int gamma_ij, int gamma_kl, double _Complex project_mn[16])
{
  auto start = target_event_create();
  auto stop = target_event_create();
  baryon_general_two_point(correl, propag_i, propag_j, propag_m, contract_type, volume, gamma_ij, gamma_kl, project_mn);
  target_event_record(start);
  target_event_synchronize(start);
  for (int i = 0; i < 100; ++i) {
    baryon_general_two_point(correl, propag_i, propag_j, propag_m, contract_type, volume, gamma_ij, gamma_kl, project_mn);
  }
  target_event_record(stop);
  target_event_synchronize(stop);
  float milliseconds = target_event_elapsed_time(start, stop);
  printf("Time elapsed: %f ms, Bandwidth: %f GB/s\n", milliseconds / 100,
         (3.0 * volume * 4 * 4 * 3 * 3 * 16 + volume * 16) / (1024 * 1024 * 1024) / (milliseconds / 1000.0 / 100));
  target_event_destory(start);
  target_event_destory(stop);

  return;
}

int main(int argc, char *argv[])
{
  init(0);

  size_t volume = 24 * 24 * 24 * 18;
  double _Complex project_mn[16];
  for (int ij = 0; ij < 16; ++ij) { project_mn[ij] = 0; }
  for (int i = 0; i < 4; ++i) { project_mn[i * 4 + i] = 1; }
  void *correl = target_malloc(volume * 2 * sizeof(double));
  void *propag_i = target_malloc(volume * 16 * 9 * 2 * sizeof(double));
  void *propag_j = target_malloc(volume * 16 * 9 * 2 * sizeof(double));
  void *propag_m = target_malloc(volume * 16 * 9 * 2 * sizeof(double));
  proton(correl, propag_i, propag_j, propag_m, IK_JL_MN, volume, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IK_JN_ML, volume, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IL_JK_MN, volume, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IL_JN_MK, volume, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IN_JK_ML, volume, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IN_JL_MK, volume, 5, 5, 0);
  proton_general(correl, propag_i, propag_j, propag_m, IK_JL_MN, volume, 5, 5, project_mn);
  proton_general(correl, propag_i, propag_j, propag_m, IK_JN_ML, volume, 5, 5, project_mn);
  proton_general(correl, propag_i, propag_j, propag_m, IL_JK_MN, volume, 5, 5, project_mn);
  proton_general(correl, propag_i, propag_j, propag_m, IL_JN_MK, volume, 5, 5, project_mn);
  proton_general(correl, propag_i, propag_j, propag_m, IN_JK_ML, volume, 5, 5, project_mn);
  proton_general(correl, propag_i, propag_j, propag_m, IN_JL_MK, volume, 5, 5, project_mn);
  proton_v2(correl, propag_i, propag_j, propag_m, IK_JL_MN, volume, 5, 5, 0);
  proton_v2(correl, propag_i, propag_j, propag_m, IK_JN_ML, volume, 5, 5, 0);
  proton_v2(correl, propag_i, propag_j, propag_m, IL_JK_MN, volume, 5, 5, 0);
  proton_v2(correl, propag_i, propag_j, propag_m, IL_JN_MK, volume, 5, 5, 0);
  proton_v2(correl, propag_i, propag_j, propag_m, IN_JK_ML, volume, 5, 5, 0);
  proton_v2(correl, propag_i, propag_j, propag_m, IN_JL_MK, volume, 5, 5, 0);
  target_free(correl);
  target_free(propag_i);
  target_free(propag_j);
  target_free(propag_m);

  return 0;
}