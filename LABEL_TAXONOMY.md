# 🏷️ 标注标签体系文档

**版本**: v1.1  
**更新时间**: 2025-12-15  
**总计**: 197 种细分标签

---

## 📊 标签概览

| 大类 | 子类数量 | 说明 |
|------|----------|------|
| 行人类 (pedestrian) | 2 | 行人和人群 |
| 车辆类 (vehicle) | 5 | 统一车辆 + 4种状态 |
| 交通标志类 (traffic_sign) | 188 | RAG 细粒度分类 |
| 施工标志类 (construction) | 2 | 交通锥和施工护栏 |

---

## 1️⃣ 行人类 (pedestrian) - 2 种

| 标签 | 说明 |
|------|------|
| `pedestrian` | 单个或少量行人 |
| `crowd` | 人群（多人聚集） |

---

## 2️⃣ 车辆类 (vehicle) - 5 种

> 统一使用 `vehicle`，只区分行驶状态，不区分车辆类型（轿车、卡车、公交等都标注为 vehicle）

| 标签 | 状态 | 判断依据 |
|------|------|----------|
| `vehicle` | 正常 | 直行或无法判断状态 |
| `vehicle_braking` | 刹车 | 尾灯明显变亮、红色刹车灯亮起 |
| `vehicle_double_flash` | 双闪 | 左右两侧转向灯同时亮起/闪烁 |
| `vehicle_turning_left` | 左转 | 车身朝左/车头转向左侧/左转车道/仅左侧转向灯亮 |
| `vehicle_turning_right` | 右转 | 车身朝右/车头转向右侧/右转车道/仅右侧转向灯亮 |

### 状态判断优先级

```
刹车 (braking) > 双闪 (double_flash) > 右转 (turning_right) > 左转 (turning_left) > 正常
```

---

## 3️⃣ 交通标志类 (traffic_sign) - 188 种

> 通过 RAG 细粒度分类识别，支持香港/中国大陆常见交通标志

### 3.1 倒计时距离牌 (Countdown Markers) - 3 种

| 标签 |
|------|
| `100m_Countdown_markers_used_to_indicate_the_distance_to_an_exit_on_the_left_side_of_a_road` |
| `200m_Countdown_markers_used_to_indicate_the_distance_to_an_exit_on_the_left_side_of_a_road` |
| `300m_Countdown_markers_used_to_indicate_the_distance_to_an_exit_on_the_left_side_of_a_road` |

### 3.2 方向/指示标志 (Direction Signs) - 8 种

| 标签 |
|------|
| `Ahead_only` |
| `Direction_in_which_the_prohibition_or_restriction_applies` |
| `Direction_sign_for_roundabout` |
| `Direction_sign` |
| `Direction_to_Mass_Transit_Railway_(MTR)_Station` |
| `Direction_to_vehicular_ferry_pier` |
| `Street_direction_sign_with_numbers` |
| `Street_direction_sign` |

### 3.3 警告标志 (Warning Signs) - 约 50 种

| 标签 |
|------|
| `Bend_to_left_ahead` |
| `Cattle_ahead` |
| `Children_ahead` |
| `Cross_roads_ahead` |
| `Cycleway_ahead_(cyclists_on_or_crossing_road_ahead)` |
| `Cyclists_ahead` |
| `Disabled_persons_ahead` |
| `Double_bend_ahead_first_to_right` |
| `Dual_carriageway_ahead` |
| `Dual_carriageway_ends_ahead` |
| `Fog_or_mist_ahead` |
| `Horses_ahead` |
| `Level_crossing_with_barrier_ahead` |
| `Light_rail_vehicles_or_trams_ahead` |
| `Loose_chippings_ahead` |
| `Low-flying_aircraft_or_noise_ahead` |
| `One_way_road_ahead` |
| `Overhead_electric_cable_ahead` |
| `Pedestrian_Accident_blackspot_ahead` |
| `Pedestrian_crossing_ahead` |
| `Pedestrian_on_or_crossing_road_ahead` |
| `Pedestrians_Ahead` |
| `Playground_ahead` |
| `Quay-side_or_river_bank_ahead` |
| `Ramp_or_sudden_change_of_road_level_ahead` |
| `Restricted_headroom_ahead` |
| `Risk_of_falling_or_fallen_rocks_ahead` |
| `Road_hump_ahead` |
| `Road_narrows_on_both_sides_ahead` |
| `Road_narrows_on_left_ahead` |
| `Road_works_ahead` |
| `School_ahead` |
| `Side_road_to_left_ahead` |
| `Slippery_road_ahead` |
| `Staggered_junction_ahead` |
| `Start_of_dual_carriageway_ahead` |
| `Steep_hill_downwards_ahead` |
| `Steep_hill_upwards_ahead` |
| `T-junction_ahead` |
| `Traffic_Accident_blackspot_ahead` |
| `Traffic_lights_ahead` |
| `Traffic_merging_from_left` |
| `Traffic_signals_ahead` |
| `Two-way_traffic_across_a_one-way_road_ahead` |
| `Two-way_traffic_ahead` |
| `Uneven_road_surface_ahead` |
| `Visually_impaired_persons_ahead` |

### 3.4 禁止标志 (Prohibition Signs) - 约 35 种

| 标签 |
|------|
| `No_bicycles` |
| `No_buses_and_coaches` |
| `No_entry_for_all_vehicles` |
| `No_entry_for_vehicles` |
| `No_exit_for_vehicles` |
| `No_goods_vehicles` |
| `No_learner_drivers` |
| `No_motor_cycles_or_motor_tricycles` |
| `No_motor_vehicles_except_motor_cyclists_and_motor_tricycles` |
| `No_motor_vehicles` |
| `No_overtaking` |
| `No_parking` |
| `No_pedestrians,_pedestrian_controlled_vehicles,_bicycles_and_tricycles` |
| `No_pedestrians,_rickshaws_and_handcarts` |
| `No_pedestrians` |
| `No_public_light_buses` |
| `No_stopping_at_any_time` |
| `No_stopping_between_7AM-12AM` |
| `No_stopping_between_7AM-7PM` |
| `No_stopping_between_8-10AM_and_5-7PM` |
| `No_stopping_for_public_light_buses_between_7AM-12AM` |
| `No_stopping` |
| `No_through_road_on_left` |
| `No_use_of_horn` |
| `No_vehicles_carrying_dangerous_goods_of_specified_categories` |
| `No_vehicles_or_combinations_of_vehicles_over_length_shown_(including_load)` |
| `No_vehicles_over_axle_weight_shown_(including_load)` |
| `No_vehicles_over_gross_vehicle_weight_shown_(including_load)` |
| `No_vehicles_over_height_shown_(including_load)` |
| `No_vehicles_over_width_shown_(including_load)` |
| `Road_ahead_closed_to_vehicles` |
| `Road_closed_to_vehicles` |

### 3.5 限速标志 (Speed Limit Signs) - 2 种

| 标签 | 说明 |
|------|------|
| `Speed_limit_(in_km_h)` | 固定限速标志 |
| `Variable_speed_limit_(in_km_h)` | 可变限速标志 |

### 3.6 停车/让行标志 (Stop/Give Way Signs) - 约 10 种

| 标签 |
|------|
| `Give_way_to_traffic_on_major_road` |
| `Stop_and_give_way` |
| `Stop_at__Census_point_` |
| `Stop_or_give_way_ahead_(with_distance_to_line_ahead_given_below)` |
| `Distance_to__Give_way__line` |
| `Distance_to__Stop__line` |
| `Prepare_to_stop_if_signalled_to_do_so` |
| `Reduce_speed_now` |

### 3.7 公交/专用道标志 (Bus Lane Signs) - 约 10 种

| 标签 |
|------|
| `Bus_lane_(Franchised_buses)_on_major_road_ahead` |
| `Bus_lane_ahead_(Franchised_buses)` |
| `Bus_lane_for_franchised_buses_only` |
| `Contra-flow_bus_lane_for_franchised_buses_only` |
| `End_of_bus_lane` |
| `Left_lane_shows_bus_lane_for_franchised_buses_only_during_the_time_and_date_shown` |

### 3.8 轻轨/电车标志 (Light Rail Signs) - 5 种

| 标签 |
|------|
| `Light_rail_vehicle_lane_or_tram_lane_ahead` |
| `Light_rail_vehicle_lane_or_tram_lane_on_major_road_ahead` |
| `Light_rail_vehicles_and_trams_only` |
| `Light_rail_vehicles_or_trams_ahead` |
| `End_of_rail_only_lane_for_light_rail_vehicles` |

### 3.9 自行车相关标志 (Bicycle Signs) - 约 10 种

| 标签 |
|------|
| `Bicycle_tricycle_route._No_motor_vehicles` |
| `Cycling_restriction_–_cyclists_must_dismount_and_push_their_cycles` |
| `Cyclists_must_dismount_and_use_crossing_to_cross_the_road` |
| `Cyclists_to_keep_left` |
| `Cyclists_to_walk_on_steep_road` |
| `End_of_cycling_restriction` |
| `Segregated_pedestrian_and_bicycle_tricycle_route._No_motor_vehicles` |

### 3.10 停车场标志 (Parking Signs) - 5 种

| 标签 |
|------|
| `Parking_place_for_buses_and_coaches_only` |
| `Parking_place_for_goods_vehicles_only` |
| `Parking_place_for_motor_cycles_only` |
| `Parking_place_for_pedal_cycles_only` |
| `Parking_place_for_vehicles_other_than_medium_and_heavy_goods_vehicles,_buses,_coaches,_motor_cycles_and_pedal_cycles` |

### 3.11 临时/施工标志 (Temporary Signs) - 约 15 种

| 标签 |
|------|
| `End_of_road_works` |
| `For_use_by_police_at_accident_site` |
| `Manually_operated_ʻStop_Goʼ_sign_ahead` |
| `Manually_operated_temporary_ʻGoʼ_sign` |
| `Manually_operated_temporary_ʻStopʼ_sign` |
| `Temporary_closure_of_lane_or_road` |
| `Temporary_closure_of_pedestrian_crossing` |
| `Temporary_route_for_pedestrians_(both_directions)` |
| `Temporary_route_for_pedestrians` |
| `Temporary_routes_for_vehicles` |
| `Temporary_sharp_deviation_to_left` |
| `Used_to_indicate_line_painting_(wording_may_be_varied_to_suit_nature_of_road_work)` |
| `Warn_of_road_surfacing_works_(wording_may_be_varied_to_suit_specific_hazard)` |

### 3.12 高速公路标志 (Expressway Signs) - 约 10 种

| 标签 |
|------|
| `End_of_an_expressway` |
| `Start_and_continuation_of_an_expressway` |
| `Hard_shoulder_–_do_not_use_except_in_an_emergency` |
| `Merging_into_main_traffic_on_left` |
| `Get_in_lane` |

### 3.13 车道指示标志 (Lane Signs) - 约 10 种

| 标签 |
|------|
| `Left_lane_only_ahead_(Red_bar_indicates_that_lanes_are_closed)` |
| `Left_lane_only_ahead_on_two-way_road_(Red_bar_indicates_that_lanes_are_closed)` |
| `Right_lane_closed_ahead_(Red_bar_indicates_that_lanes_are_closed)` |
| `One_way_traffic` |
| `Vehicles_must_use_the_left_most_lane_except_when_overtaking` |
| `Vehicles_must_use_the_right_most_lane_except_when_overtaking` |

### 3.14 其他标志 (Other Signs) - 约 35 种

| 标签 |
|------|
| `Census_point` |
| `Day_plate` |
| `Time_plate` |
| `Keep_in_low_gear` |
| `Keep_right_(keep_left_if_symbol_reversed)` |
| `Passing_place` |
| `Pedestrian_priority_zone` |
| `Red_light_camera_control_zone` |
| `Red_light_speed_camera_ahead` |
| `Sharp_deviation_of_route_to_left` |
| `Sign_at_start_of_single_track_road` |
| `Single_file_traffic_ahead` |
| `Slow_(Sign_used_by_police_in_emergency)` |
| `Use_low_gear_for_distance_shown` |
| `Use_low_gear` |
| `Vehicles_must_stop_at_the_sign_(sign_used_by_police)` |
| `Vehicles_must_stop_at_the_sign_(sign_used_by_school_crossing_patrol)` |
| `Way_in_for_vehicles` |
| `Way_out_for_vehicles` |
| ... |

---

## 4️⃣ 施工标志类 (construction) - 2 种

| 标签 | 说明 |
|------|------|
| `traffic_cone` | 交通锥 |
| `construction_barrier` | 施工护栏 |

---

## 📋 标签统计汇总

```
总计: 197 种标签

├── 行人类 (pedestrian): 2 种
│   ├── pedestrian
│   └── crowd
│
├── 车辆类 (vehicle): 5 种
│   ├── vehicle（正常）
│   ├── vehicle_braking（刹车）
│   ├── vehicle_double_flash（双闪）
│   ├── vehicle_turning_left（左转）
│   └── vehicle_turning_right（右转）
│
├── 交通标志类 (traffic_sign): 188 种
│   ├── 倒计时距离牌: 3 种
│   ├── 方向/指示标志: 8 种
│   ├── 警告标志: ~50 种
│   ├── 禁止标志: ~35 种
│   ├── 限速标志: 2 种
│   ├── 停车/让行标志: ~10 种
│   ├── 公交/专用道标志: ~10 种
│   ├── 轻轨/电车标志: 5 种
│   ├── 自行车相关标志: ~10 种
│   ├── 停车场标志: 5 种
│   ├── 临时/施工标志: ~15 种
│   ├── 高速公路标志: ~10 种
│   ├── 车道指示标志: ~10 种
│   └── 其他标志: ~35 种
│
└── 施工标志类 (construction): 2 种
    ├── traffic_cone
    └── construction_barrier
```

---

## 🔧 备注

1. **交通标志的细粒度分类**通过 RAG (Retrieval-Augmented Generation) 技术实现，使用二阶段识别流程
2. **车辆状态判断优先级**: 刹车 > 双闪 > 右转 > 左转 > 正常
3. 所有机动车（轿车、卡车、公交、摩托车等）和自行车都统一标注为 `vehicle`，只区分状态
4. 所有标签均为小写，使用下划线连接

---

*此文档由 GLM_Labeling 项目自动生成*
