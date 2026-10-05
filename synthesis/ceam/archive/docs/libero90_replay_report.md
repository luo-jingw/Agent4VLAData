# LIBERO-90 CPU regeneration report

This report records the full deterministic CPU regeneration completed on 2026-09-30. The raw hdf5 files are read-only. A demo is retained only when `done[-1]` is true.

## Dataset identity

| Item | Measured value |
|---|---:|
| benchmark task identities | 90 |
| scenes | 20 |
| original instruction identities | 74 |
| raw demos | 4500 |
| retained episodes | 3949 |
| retained frames | 572988 |
| longest raw-filtered demo, all / retained | 370 / 370 |
| zero-retained task | `LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` |
| replay joint maximum absolute difference from raw observations | 2.49e+00 |
| replay gripper maximum absolute difference from raw observations | 5.14e-02 |

The assembled LeRobot v2.1 dataset is `outputs/datasets/ceam_regenerated/libero_90_no_noops`. `task_index` represents the 74 instruction identities. `meta/evaluation_tasks.jsonl` represents all 90 benchmark task identities, including the zero-retained task. `meta/episode_sources.jsonl` maps each assembled episode to its benchmark task, scene, raw file and raw demo.

## Resource use

| Item | Measured value |
|---|---:|
| nodes | node-61, node-63, node-65, node-66 |
| CPU workers | 8, two per node |
| worker wall time, maximum | 1277.91 s |
| replay report / assembly / exact validation wall | 2.87 / 12.94 / 20.61 s |
| measured CPU time, user / system / total | 9053.59 / 113.07 / 9166.66 s |
| per-worker peak RSS, min / max | 3.22 / 3.61 GiB |
| GPU allocation | 0 |
| regeneration records | 820.4 MiB |
| assembled dataset | 101.8 MiB |

Each worker used `MUJOCO_GL=disable` and one thread for OMP, MKL, OpenBLAS and NUMBA. The eight workers were split evenly over node-61, node-63, node-65 and node-66.

## Per-task replay measurements

The joint and gripper columns compare replay observations with the raw hdf5 observation arrays at the no-op-filtered steps. Object distances compare replay simulator positions with the raw simulator states. Distances are millimetres.

| Benchmark task | Scene | Kept | Retained frames | Joint median/max | Gripper max | Moved object median/p90/max | Static object max | Seconds/demo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet` | KITCHEN_SCENE10 | 48/50 | 3605 | 3.43e-02/1.26e-01 | 2.59e-03 | - | 0.3 | 1.1 |
| `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it` | KITCHEN_SCENE10 | 44/50 | 9186 | 3.67e-02/2.92e-01 | 2.09e-02 | 11.1/81.8/401.3 | 407.6 | 2.9 |
| `KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | KITCHEN_SCENE10 | 50/50 | 6185 | 5.59e-02/1.60e+00 | 1.49e-02 | 7.7/18.5/66.5 | 14.8 | 1.7 |
| `KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it` | KITCHEN_SCENE10 | 48/50 | 9017 | 3.72e-02/1.13e+00 | 2.57e-02 | 3.8/12.5/63.0 | 21.8 | 2.5 |
| `KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it` | KITCHEN_SCENE10 | 48/50 | 8609 | 4.42e-02/1.97e-01 | 1.95e-02 | 4.3/11.0/115.3 | 22.2 | 2.4 |
| `KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it` | KITCHEN_SCENE10 | 47/50 | 8369 | 4.26e-02/6.12e-01 | 4.00e-02 | 4.2/14.3/266.9 | 20.2 | 2.4 |
| `KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet` | KITCHEN_SCENE1 | 46/50 | 6526 | 6.32e-02/4.05e-01 | 3.28e-02 | 25.2/56.6/105.8 | 40.4 | 2.0 |
| `KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet` | KITCHEN_SCENE1 | 43/50 | 3865 | 3.63e-02/5.96e-01 | 1.44e-02 | - | 0.2 | 1.2 |
| `KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it` | KITCHEN_SCENE1 | 38/50 | 7471 | 4.56e-02/1.42e+00 | 3.31e-02 | 8.6/27.5/361.3 | 360.0 | 2.8 |
| `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate` | KITCHEN_SCENE1 | 46/50 | 6005 | 5.32e-02/6.04e-01 | 9.89e-03 | 6.4/18.4/41.7 | 19.7 | 1.7 |
| `KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet` | KITCHEN_SCENE1 | 50/50 | 6783 | 4.07e-02/2.17e-01 | 1.31e-02 | 8.6/23.8/71.8 | 13.4 | 1.8 |
| `KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet` | KITCHEN_SCENE2 | 49/50 | 3674 | 3.52e-02/3.06e-01 | 1.08e-02 | - | 0.2 | 1.2 |
| `KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate` | KITCHEN_SCENE2 | 46/50 | 5081 | 6.22e-02/2.98e-01 | 1.41e-02 | 5.9/19.6/110.8 | 12.2 | 1.9 |
| `KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate` | KITCHEN_SCENE2 | 44/50 | 6063 | 5.03e-02/1.79e+00 | 2.20e-02 | 10.7/25.2/180.9 | 24.6 | 2.4 |
| `KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate` | KITCHEN_SCENE2 | 49/50 | 5226 | 3.78e-02/1.21e-01 | 2.41e-02 | 6.1/17.3/50.9 | 11.7 | 2.0 |
| `KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet` | KITCHEN_SCENE2 | 48/50 | 6531 | 4.15e-02/1.88e-01 | 2.04e-02 | 9.7/20.2/532.5 | 26.0 | 2.5 |
| `KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle` | KITCHEN_SCENE2 | 48/50 | 6144 | 4.99e-02/2.85e-01 | 1.89e-02 | 8.2/25.1/145.3 | 17.1 | 2.3 |
| `KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl` | KITCHEN_SCENE2 | 42/50 | 4759 | 3.67e-02/2.20e-01 | 1.76e-02 | 8.7/20.6/74.1 | 24.0 | 2.0 |
| `KITCHEN_SCENE3_put_the_frying_pan_on_the_stove` | KITCHEN_SCENE3 | 47/50 | 9264 | 4.75e-02/3.76e-01 | 1.83e-02 | 11.4/31.8/271.0 | 21.4 | 2.2 |
| `KITCHEN_SCENE3_put_the_moka_pot_on_the_stove` | KITCHEN_SCENE3 | 47/50 | 7029 | 5.11e-02/1.15e+00 | 3.90e-02 | 6.1/21.5/221.6 | 284.8 | 1.7 |
| `KITCHEN_SCENE3_turn_on_the_stove` | KITCHEN_SCENE3 | 50/50 | 4513 | 5.07e-02/3.47e-01 | 3.46e-02 | - | 0.2 | 1.1 |
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it` | KITCHEN_SCENE3 | 31/50 | 8522 | 5.70e-02/4.60e-01 | 3.64e-02 | 20.4/254.4/552.0 | 500.5 | 3.1 |
| `KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet` | KITCHEN_SCENE4 | 42/50 | 4840 | 8.10e-02/8.36e-01 | 2.35e-02 | 19.1/81.2/149.8 | 166.6 | 1.6 |
| `KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer` | KITCHEN_SCENE4 | 32/50 | 7456 | 6.28e-02/7.66e-01 | 2.43e-02 | 22.5/43.4/45.3 | 155.8 | 3.0 |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet` | KITCHEN_SCENE4 | 45/50 | 5865 | 4.84e-02/3.32e-01 | 2.53e-02 | 9.5/27.9/238.0 | 25.1 | 1.6 |
| `KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet` | KITCHEN_SCENE4 | 48/50 | 7539 | 4.54e-02/9.62e-01 | 2.60e-02 | 8.9/25.9/448.0 | 42.2 | 1.9 |
| `KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet` | KITCHEN_SCENE4 | 41/50 | 5122 | 6.86e-02/9.39e-01 | 1.23e-02 | 7.2/47.9/330.2 | 314.3 | 1.6 |
| `KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack` | KITCHEN_SCENE4 | 38/50 | 9161 | 6.02e-02/6.49e-01 | 3.82e-02 | 8.4/29.4/291.3 | 26.8 | 2.9 |
| `KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet` | KITCHEN_SCENE5 | 50/50 | 3715 | 3.45e-02/2.09e-01 | 6.08e-03 | - | 1.4 | 1.0 |
| `KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | KITCHEN_SCENE5 | 50/50 | 6090 | 4.12e-02/1.32e+00 | 2.76e-02 | 6.3/21.3/146.6 | 18.0 | 1.6 |
| `KITCHEN_SCENE5_put_the_black_bowl_on_the_plate` | KITCHEN_SCENE5 | 44/50 | 5900 | 4.44e-02/6.05e-01 | 1.16e-02 | 7.0/16.2/43.3 | 15.1 | 2.0 |
| `KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet` | KITCHEN_SCENE5 | 50/50 | 7545 | 4.62e-02/1.88e+00 | 9.63e-03 | 7.3/20.1/168.2 | 19.7 | 2.2 |
| `KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet` | KITCHEN_SCENE5 | 43/50 | 8995 | 4.20e-02/5.03e-01 | 3.38e-02 | 4.9/20.9/363.1 | 333.3 | 2.7 |
| `KITCHEN_SCENE6_close_the_microwave` | KITCHEN_SCENE6 | 46/50 | 9214 | 4.45e-02/1.55e+00 | 4.08e-02 | 7.5/17.5/114.1 | 240.9 | 2.4 |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug` | KITCHEN_SCENE6 | 44/50 | 5493 | 4.67e-02/6.44e-01 | 1.92e-02 | 8.4/20.8/71.4 | 17.4 | 1.5 |
| `KITCHEN_SCENE7_open_the_microwave` | KITCHEN_SCENE7 | 46/50 | 6805 | 5.29e-02/2.49e+00 | 2.61e-02 | 54.1/54.1/54.1 | 6.0 | 1.5 |
| `KITCHEN_SCENE7_put_the_white_bowl_on_the_plate` | KITCHEN_SCENE7 | 43/50 | 8224 | 3.94e-02/1.89e-01 | 1.13e-02 | 7.8/22.0/286.2 | 321.2 | 2.4 |
| `KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate` | KITCHEN_SCENE7 | 44/50 | 6014 | 3.75e-02/3.22e-01 | 2.77e-02 | 9.4/79.9/459.8 | 18.6 | 1.7 |
| `KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove` | KITCHEN_SCENE8 | 40/50 | 8332 | 3.47e-02/2.77e-01 | 3.69e-02 | 9.7/65.3/437.3 | 475.1 | 2.7 |
| `KITCHEN_SCENE8_turn_off_the_stove` | KITCHEN_SCENE8 | 44/50 | 7607 | 6.76e-02/1.66e+00 | 4.05e-02 | - | 0.0 | 2.4 |
| `KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf` | KITCHEN_SCENE9 | 36/50 | 6710 | 4.53e-02/4.82e-01 | 2.90e-02 | 12.4/55.0/187.3 | 16.9 | 2.0 |
| `KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet` | KITCHEN_SCENE9 | 40/50 | 7341 | 6.34e-02/1.06e+00 | 3.69e-02 | 13.9/65.7/390.4 | 377.2 | 1.9 |
| `KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf` | KITCHEN_SCENE9 | 45/50 | 7545 | 4.28e-02/1.02e+00 | 2.66e-02 | 8.1/23.8/287.0 | 259.5 | 1.8 |
| `KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet` | KITCHEN_SCENE9 | 45/50 | 6661 | 4.26e-02/6.30e-01 | 2.96e-02 | 7.8/27.4/481.0 | 13.7 | 1.8 |
| `KITCHEN_SCENE9_turn_on_the_stove` | KITCHEN_SCENE9 | 49/50 | 5720 | 3.04e-02/4.58e-01 | 2.88e-02 | - | 0.2 | 1.3 |
| `KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it` | KITCHEN_SCENE9 | 41/50 | 10568 | 6.23e-02/1.16e+00 | 3.80e-02 | 14.4/41.2/310.1 | 364.1 | 2.7 |
| `LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket` | LIVING_ROOM_SCENE1 | 50/50 | 6936 | 4.60e-02/1.84e-01 | 3.42e-02 | 3.0/11.2/121.8 | 122.1 | 1.9 |
| `LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket` | LIVING_ROOM_SCENE1 | 47/50 | 6912 | 3.34e-02/1.23e+00 | 2.03e-02 | 3.6/25.7/135.6 | 122.1 | 2.0 |
| `LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket` | LIVING_ROOM_SCENE1 | 47/50 | 7984 | 3.27e-02/2.82e-01 | 2.07e-02 | 5.0/28.3/456.8 | 122.1 | 2.2 |
| `LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | LIVING_ROOM_SCENE1 | 46/50 | 7245 | 3.16e-02/1.94e-01 | 3.48e-02 | 3.4/17.4/497.9 | 477.4 | 2.0 |
| `LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket` | LIVING_ROOM_SCENE2 | 49/50 | 6860 | 4.16e-02/6.27e-01 | 2.50e-02 | 4.4/21.6/118.2 | 122.1 | 2.2 |
| `LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` | LIVING_ROOM_SCENE2 | 0/50 | 0 | 4.13e-02/2.24e-01 | 2.42e-02 | 11.1/11.8/11.8 | 306.7 | 1.8 |
| `LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket` | LIVING_ROOM_SCENE2 | 43/50 | 4989 | 3.45e-02/3.48e-01 | 3.63e-02 | 3.3/17.7/412.6 | 395.7 | 1.8 |
| `LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket` | LIVING_ROOM_SCENE2 | 40/50 | 6467 | 3.85e-02/3.40e-01 | 3.39e-02 | 5.1/34.3/543.5 | 517.2 | 2.5 |
| `LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | LIVING_ROOM_SCENE2 | 49/50 | 5159 | 3.84e-02/4.48e-01 | 3.32e-02 | 3.5/26.8/155.8 | 165.9 | 1.9 |
| `LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` | LIVING_ROOM_SCENE3 | 47/50 | 5605 | 3.77e-02/5.88e-01 | 3.35e-02 | 3.3/14.2/363.8 | 410.8 | 1.9 |
| `LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray` | LIVING_ROOM_SCENE3 | 50/50 | 5027 | 3.29e-02/1.63e-01 | 1.85e-02 | 3.1/8.2/101.7 | 122.1 | 1.4 |
| `LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray` | LIVING_ROOM_SCENE3 | 48/50 | 7289 | 3.57e-02/2.12e-01 | 2.46e-02 | 4.8/17.5/472.1 | 458.9 | 2.0 |
| `LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray` | LIVING_ROOM_SCENE3 | 48/50 | 7571 | 3.55e-02/3.01e-01 | 1.85e-02 | 4.3/12.4/428.5 | 122.1 | 2.1 |
| `LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray` | LIVING_ROOM_SCENE3 | 48/50 | 5258 | 3.25e-02/4.63e-01 | 2.07e-02 | 2.5/24.2/271.9 | 122.1 | 1.6 |
| `LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray` | LIVING_ROOM_SCENE4 | 49/50 | 5984 | 4.23e-02/3.68e-01 | 3.63e-02 | 7.7/21.7/239.9 | 20.2 | 1.8 |
| `LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray` | LIVING_ROOM_SCENE4 | 48/50 | 7312 | 3.35e-02/1.52e-01 | 2.51e-02 | 3.5/12.5/541.3 | 501.9 | 2.2 |
| `LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray` | LIVING_ROOM_SCENE4 | 46/50 | 5337 | 2.96e-02/1.64e-01 | 2.03e-02 | 2.9/13.1/389.8 | 472.0 | 2.0 |
| `LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray` | LIVING_ROOM_SCENE4 | 48/50 | 10259 | 4.20e-02/3.83e-01 | 2.69e-02 | 9.7/31.0/236.7 | 20.2 | 3.9 |
| `LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray` | LIVING_ROOM_SCENE4 | 38/50 | 8878 | 4.60e-02/4.82e-01 | 3.32e-02 | 13.6/50.2/466.6 | 428.2 | 3.9 |
| `LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate` | LIVING_ROOM_SCENE5 | 37/50 | 5246 | 2.73e-02/2.97e-01 | 1.94e-02 | 9.2/27.1/355.4 | 24.5 | 2.2 |
| `LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate` | LIVING_ROOM_SCENE5 | 44/50 | 6148 | 3.51e-02/3.92e-01 | 3.21e-02 | 8.7/18.6/35.7 | 379.1 | 2.2 |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate` | LIVING_ROOM_SCENE5 | 44/50 | 4707 | 3.95e-02/3.64e-01 | 3.71e-02 | 6.4/19.0/40.4 | 17.5 | 1.7 |
| `LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate` | LIVING_ROOM_SCENE5 | 47/50 | 5004 | 3.01e-02/2.07e-01 | 1.34e-02 | 7.6/20.3/49.8 | 45.3 | 1.7 |
| `LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate` | LIVING_ROOM_SCENE6 | 48/50 | 5972 | 3.33e-02/2.02e-01 | 2.68e-02 | 4.2/14.0/349.0 | 252.7 | 1.6 |
| `LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate` | LIVING_ROOM_SCENE6 | 50/50 | 4447 | 3.68e-02/1.75e-01 | 1.71e-02 | 2.6/8.1/47.1 | 17.0 | 1.3 |
| `LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate` | LIVING_ROOM_SCENE6 | 44/50 | 5647 | 3.31e-02/3.02e-01 | 2.61e-02 | 7.4/21.0/393.9 | 32.7 | 1.9 |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate` | LIVING_ROOM_SCENE6 | 44/50 | 6897 | 3.48e-02/4.26e-01 | 2.42e-02 | 8.0/21.1/348.3 | 21.7 | 2.0 |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` | STUDY_SCENE1 | 34/50 | 5982 | 4.14e-02/6.59e-01 | 2.81e-02 | 4.3/24.8/384.1 | 13.7 | 1.9 |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` | STUDY_SCENE1 | 44/50 | 6949 | 4.31e-02/2.03e-01 | 2.60e-02 | 4.4/12.4/153.9 | 20.4 | 1.7 |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` | STUDY_SCENE1 | 31/50 | 4883 | 5.59e-02/5.33e-01 | 5.14e-02 | 5.3/29.1/198.7 | 333.9 | 1.6 |
| `STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy` | STUDY_SCENE1 | 45/50 | 5682 | 4.15e-02/3.76e-01 | 1.08e-02 | 8.7/16.2/137.1 | 17.9 | 1.4 |
| `STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | STUDY_SCENE2 | 44/50 | 6438 | 4.33e-02/3.91e-01 | 1.85e-02 | 4.2/11.9/91.6 | 16.1 | 1.5 |
| `STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` | STUDY_SCENE2 | 30/50 | 4402 | 4.33e-02/3.56e-01 | 2.94e-02 | 4.1/35.6/185.1 | 10.5 | 1.7 |
| `STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` | STUDY_SCENE2 | 48/50 | 5846 | 3.69e-02/2.66e-01 | 2.29e-02 | 3.5/11.1/67.8 | 15.3 | 1.4 |
| `STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` | STUDY_SCENE2 | 34/50 | 4959 | 4.14e-02/5.40e-01 | 2.38e-02 | 4.5/35.7/153.0 | 18.9 | 1.5 |
| `STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` | STUDY_SCENE3 | 26/50 | 4493 | 4.28e-02/7.45e-01 | 3.90e-02 | 5.2/44.6/122.0 | 11.2 | 2.1 |
| `STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` | STUDY_SCENE3 | 46/50 | 6691 | 3.80e-02/6.72e-01 | 3.36e-02 | 4.0/14.0/135.1 | 16.5 | 1.7 |
| `STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` | STUDY_SCENE3 | 32/50 | 4414 | 4.74e-02/4.07e-01 | 3.13e-02 | 5.2/25.4/155.8 | 11.4 | 1.6 |
| `STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy` | STUDY_SCENE3 | 45/50 | 5318 | 6.05e-02/3.99e-01 | 2.10e-02 | 9.3/23.2/44.8 | 419.1 | 1.4 |
| `STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy` | STUDY_SCENE3 | 47/50 | 4808 | 3.81e-02/2.65e-01 | 2.66e-02 | 6.0/13.9/33.2 | 14.4 | 1.3 |
| `STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf` | STUDY_SCENE4 | 49/50 | 7906 | 4.40e-02/3.05e-01 | 2.77e-02 | 4.4/9.4/89.7 | 14.4 | 1.8 |
| `STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf` | STUDY_SCENE4 | 46/50 | 7820 | 4.47e-02/4.49e-01 | 4.00e-02 | 5.3/17.3/616.1 | 45.0 | 2.0 |
| `STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf` | STUDY_SCENE4 | 47/50 | 4774 | 4.27e-02/4.08e-01 | 2.85e-02 | 4.2/17.1/166.8 | 26.9 | 1.1 |
| `STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf` | STUDY_SCENE4 | 47/50 | 5594 | 4.45e-02/6.39e-01 | 3.06e-02 | 5.3/18.4/106.8 | 22.4 | 1.3 |
