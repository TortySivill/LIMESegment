

TRAIN_FILES = ['Data/Coffee_TRAIN', #0
                'Data/Strawberry_TRAIN', #1
                'Data/GunPointOldVersusYoung_TRAIN', #2
                'Data/HandOutlines_TRAIN', #3
                'Data/Yoga_TRAIN', #4
                'Data/ECG200_TRAIN', #5
                'Data/GunPointMaleVersusFemale_TRAIN', #6
                'Data/DodgerLoopGame_TRAIN', #7
                'Data/Chinatown_TRAIN', #8
                'Data/FreezerSmallTrain_TRAIN', #9
                'Data/HouseTwenty_TRAIN', #10
                'Data/WormsTwoClass_TRAIN', #11
            ]
TEST_FILES = ['Data/Coffee_TEST', #0
                'Data/Strawberry_TEST', #1
                'Data/GunPointOldVersusYoung_TEST', #2
                'Data/HandOutlines_TEST', #3
                'Data/Yoga_TEST', #4
                'Data/ECG200_TEST', #5
                'Data/GunPointMaleVersusFemale_TEST', #6
                'Data/DodgerLoopGame_TEST', #7
                'Data/Chinatown_TEST', #8
                'Data/FreezerSmallTrain_TEST', #9
                'Data/HouseTwenty_TEST', #10
                'Data/WormsTwoClass_TEST', #11
            ]

TRAIN1_FILES = ['../data//Adiac_TRAIN',  # 0
               '../data//ArrowHead_TRAIN',  # 1
               '../data//ChlorineConcentration_TRAIN',  # 2
               '../data//InsectWingbeatSound_TRAIN',  # 3
               '../data//Lighting7_TRAIN',  # 4
               '../data//Wine_TRAIN',  # 5
               '../data//WordsSynonyms_TRAIN',  # 6
               '../data//50words_TRAIN',  # 7
               '../data//Beef_TRAIN',  # 8
               '../data//DistalPhalanxOutlineAgeGroup_TRAIN',  # 9
               '../data//DistalPhalanxOutlineCorrect_TRAIN',  # 10
               '../data//DistalPhalanxTW_TRAIN',  # 11
               '../data//ECG200_TRAIN',  # 12
               '../data//ECGFiveDays_TRAIN',  # 13
               '../data//BeetleFly_TRAIN',  # 14
               '../data//BirdChicken_TRAIN',  # 15
               '../data//ItalyPowerDemand_TRAIN',  # 16
               '../data//SonyAIBORobotSurface_TRAIN',  # 17
               '../data//SonyAIBORobotSurfaceII_TRAIN',  # 18
               '../data//MiddlePhalanxOutlineAgeGroup_TRAIN',  # 19
               '../data//MiddlePhalanxOutlineCorrect_TRAIN',  # 20
               '../data//MiddlePhalanxTW_TRAIN',  # 21
               '../data//ProximalPhalanxOutlineAgeGroup_TRAIN',  # 22
               '../data//ProximalPhalanxOutlineCorrect_TRAIN',  # 23
               '../data//ProximalPhalanxTW_TRAIN',  # 24
               '../data//MoteStrain_TRAIN',  # 25
               '../data//MedicalImages_TRAIN',  # 26
               '../data//Strawberry_TRAIN',  # 27
               '../data//ToeSegmentation1_TRAIN',  # 28
               'Data/Coffee_TRAIN',  # 29
               '../data//Cricket_X_TRAIN',  # 30
               '../data//Cricket_Y_TRAIN',  # 31
               '../data//Cricket_Z_TRAIN',  # 32
               '../data//uWaveGestureLibrary_X_TRAIN',  # 33
               '../data//uWaveGestureLibrary_Y_TRAIN',  # 34
               '../data//uWaveGestureLibrary_Z_TRAIN',  # 35
               '../data//ToeSegmentation2_TRAIN',  # 36
               '../data//DiatomSizeReduction_TRAIN',  # 37
               '../data//Car_TRAIN',  # 38
               '../data//CBF_TRAIN',  # 39
               '../data//CinC_ECG_torso_TRAIN',  # 40
               '../data//Computers_TRAIN',  # 41
               '../data//Earthquakes_TRAIN',  # 42
               '../data//ECG5000_TRAIN',  # 43
               '../data//ElectricDevices_TRAIN',  # 44
               '../data//FaceAll_TRAIN',  # 45
               '../data//FaceFour_TRAIN',  # 46
               '../data//FacesUCR_TRAIN',  # 47
               '../data//Fish_TRAIN',  # 48
               '../data//FordA_TRAIN',  # 49
               '../data//FordB_TRAIN',  # 50
               '../data//Gun_Point_TRAIN',  # 51
               '../data//Ham_TRAIN',  # 52
               '../data//HandOutlines_TRAIN',  # 53
               '../data//Haptics_TRAIN',  # 54
               '../data//Herring_TRAIN',  # 55
               '../data//InlineSkate_TRAIN',  # 56
               '../data//LargeKitchenAppliances_TRAIN',  # 57
               '../data//Lighting2_TRAIN',  # 58
               '../data//Mallat_TRAIN',  # 59
               '../data//Meat_TRAIN',  # 60
               '../data//NonInvasiveFatalECG_Thorax1_TRAIN',  # 61
               '../data//NonInvasiveFatalECG_Thorax2_TRAIN',  # 62
               '../data//OliveOil_TRAIN',  # 63
               '../data//OSULeaf_TRAIN',  # 64
               '../data//PhalangesOutlinesCorrect_TRAIN',  # 65
               '../data//Phoneme_TRAIN',  # 66
               '../data//Plane_TRAIN',  # 67
               '../data//RefrigerationDevices_TRAIN',  # 68
               '../data//ScreenType_TRAIN',  # 69
               '../data//ShapeletSim_TRAIN',  # 70
               '../data//ShapesAll_TRAIN',  # 71
               '../data//SmallKitchenAppliances_TRAIN',  # 72
               '../data//StarLightCurves_TRAIN',  # 73
               '../data//SwedishLeaf_TRAIN',  # 74
               '../data//Symbols_TRAIN',  # 75
               '../data//synthetic_control_TRAIN',  # 76
               '../data//Trace_TRAIN',  # 77
               '../data//Two_Patterns_TRAIN',  # 78
               '../data//TwoLeadECG_TRAIN',  # 79
               '../data//UWaveGestureLibraryAll_TRAIN',  # 80
               '../data//Wafer_TRAIN',  # 81
               '../data//Worms_TRAIN',  # 82
               '../data//WormsTwoClass_TRAIN',  # 83
               '../data//Yoga_TRAIN',  # 84
               '../data//ACSF1_TRAIN',  # 85
               '../data//AllGestureWiimoteX_TRAIN',  # 86
               '../data//AllGestureWiimoteY_TRAIN',  # 87
               '../data//AllGestureWiimoteZ_TRAIN',  # 88
               '../data//BME_TRAIN',  # 89
               '../data//Chinatown_TRAIN',  # 90
               '../data//Crop_TRAIN',  # 91
               '../data//DodgerLoopDay_TRAIN',  # 92
               '../data//DodgerLoopGame_TRAIN',  # 93
               '../data//DodgerLoopWeekend_TRAIN',  # 94
               '../data//EOGHorizontalSignal_TRAIN',  # 95
               '../data//EOGVerticalSignal_TRAIN',  # 96
               '../data//EthanolLevel_TRAIN',  # 97
               '../data//FreezerRegularTrain_TRAIN',  # 98
               '../data//FreezerSmallTrain_TRAIN',  # 99
               '../data//Fungi_TRAIN',  # 100
               '../data//GestureMidAirD1_TRAIN',  # 101
               '../data//GestureMidAirD2_TRAIN',  # 102
               '../data//GestureMidAirD3_TRAIN',  # 103
               '../data//GesturePebbleZ1_TRAIN',  # 104
               '../data//GesturePebbleZ2_TRAIN',  # 105
               '../data//GunPointAgeSpan_TRAIN',  # 106
               '../data//GunPointMaleVersusFemale_TRAIN',  # 107
               '../data//GunPointOldVersusYoung_TRAIN',  # 108
               '../data//HouseTwenty_TRAIN',  # 109
               '../data//InsectEPGRegularTrain_TRAIN',  # 110
               '../data//InsectEPGSmallTrain_TRAIN',  # 111
               '../data//MelbournePedestrian_TRAIN',  # 112
               '../data//MixedShapesRegularTrain_TRAIN',  # 113
               '../data//MixedShapesSmallTrain_TRAIN',  # 114
               '../data//PickupGestureWiimoteZ_TRAIN',  # 115
               '../data//PigAirwayPressure_TRAIN',  # 116
               '../data//PigArtPressure_TRAIN',  # 117
               '../data//PigCVP_TRAIN',  # 118
               '../data//PLAID_TRAIN',  # 119
               '../data//PowerCons_TRAIN',  # 120
               '../data//Rock_TRAIN',  # 121
               '../data//SemgHandGenderCh2_TRAIN',  # 122
               '../data//SemgHandMovementCh2_TRAIN',  # 123
               '../data//SemgHandSubjectCh2_TRAIN',  # 124
               '../data//ShakeGestureWiimoteZ_TRAIN',  # 125
               '../data//SmoothSubspace_TRAIN',  # 126
               '../data//UMD_TRAIN',  # 127
               '../data//Synthetic_TRAIN.txt', #128
               '../data//Depresjon_TRAIN.txt', #129

               ]

TEST_FILES1 = ['../data//Adiac_TEST',  # 0
              '../data//ArrowHead_TEST',  # 1
              '../data//ChlorineConcentration_TEST',  # 2
              '../data//InsectWingbeatSound_TEST',  # 3
              '../data//Lighting7_TEST',  # 4
              '../data//Wine_TEST',  # 5
              '../data//WordsSynonyms_TEST',  # 6
              '../data//50words_TEST',  # 7
              '../data//Beef_TEST',  # 8
              '../data//DistalPhalanxOutlineAgeGroup_TEST',  # 9
              '../data//DistalPhalanxOutlineCorrect_TEST',  # 10
              '../data//DistalPhalanxTW_TEST',  # 11
              '../data//ECG200_TEST',  # 12
              '../data//ECGFiveDays_TEST',  # 13
              '../data//BeetleFly_TEST',  # 14
              '../data//BirdChicken_TEST',  # 15
              '../data//ItalyPowerDemand_TEST',  # 16
              '../data//SonyAIBORobotSurface_TEST',  # 17
              '../data//SonyAIBORobotSurfaceII_TEST',  # 18
              '../data//MiddlePhalanxOutlineAgeGroup_TEST',  # 19 (inverted dataset)
              '../data//MiddlePhalanxOutlineCorrect_TEST',  # 20 (inverted dataset)
              '../data//MiddlePhalanxTW_TEST',  # 21 (inverted dataset)
              '../data//ProximalPhalanxOutlineAgeGroup_TEST',  # 22
              '../data//ProximalPhalanxOutlineCorrect_TEST',  # 23
              '../data//ProximalPhalanxTW_TEST',  # 24 (inverted dataset)
              '../data//MoteStrain_TEST',  # 25
              '../data//MedicalImages_TEST',  # 26
              '../data//Strawberry_TEST',  # 27
              '../data//ToeSegmentation1_TEST',  # 28
              'Data/Coffee_TEST',  # 29
              '../data//Cricket_X_TEST',  # 30
              '../data//Cricket_Y_TEST',  # 31
              '../data//Cricket_Z_TEST',  # 32
              '../data//uWaveGestureLibrary_X_TEST',  # 33
              '../data//uWaveGestureLibrary_Y_TEST',  # 34
              '../data//uWaveGestureLibrary_Z_TEST',  # 35
              '../data//ToeSegmentation2_TEST',  # 36
              '../data//DiatomSizeReduction_TEST',  # 37
              '../data//Car_TEST',  # 38
              '../data//CBF_TEST',  # 39
              '../data//CinC_ECG_torso_TEST',  # 40
              '../data//Computers_TEST',  # 41
              '../data//Earthquakes_TEST',  # 42
              '../data//ECG5000_TEST',  # 43
              '../data//ElectricDevices_TEST',  # 44
              '../data//FaceAll_TEST',  # 45
              '../data//FaceFour_TEST',  # 46
              '../data//FacesUCR_TEST',  # 47
              '../data//Fish_TEST',  # 48
              '../data//FordA_TEST',  # 49
              '../data//FordB_TEST',  # 50
              '../data//Gun_Point_TEST',  # 51
              '../data//Ham_TEST',  # 52
              '../data//HandOutlines_TEST',  # 53
              '../data//Haptics_TEST',  # 54
              '../data//Herring_TEST',  # 55
              '../data//InlineSkate_TEST',  # 56
              '../data//LargeKitchenAppliances_TEST',  # 57
              '../data//Lighting2_TEST',  # 58
              '../data//Mallat_TEST',  # 59
              '../data//Meat_TEST',  # 60
              '../data//NonInvasiveFatalECG_Thorax1_TEST',  # 61
              '../data//NonInvasiveFatalECG_Thorax2_TEST',  # 62
              '../data//OliveOil_TEST',  # 63
              '../data//OSULeaf_TEST',  # 64
              '../data//PhalangesOutlinesCorrect_TEST',  # 65
              '../data//Phoneme_TEST',  # 66
              '../data//Plane_TEST',  # 67
              '../data//RefrigerationDevices_TEST',  # 68
              '../data//ScreenType_TEST',  # 69
              '../data//ShapeletSim_TEST',  # 70
              '../data//ShapesAll_TEST',  # 71
              '../data//SmallKitchenAppliances_TEST',  # 72
              '../data//StarLightCurves_TEST',  # 73
              '../data//SwedishLeaf_TEST',  # 74
              '../data//Symbols_TEST',  # 75
              '../data//synthetic_control_TEST',  # 76
              '../data//Trace_TEST',  # 77
              '../data//Two_Patterns_TEST',  # 78
              '../data//TwoLeadECG_TEST',  # 79
              '../data//UWaveGestureLibraryAll_TEST',  # 80
              '../data//Wafer_TEST',  # 81
              '../data//Worms_TEST',  # 82
              '../data//WormsTwoClass_TEST',  # 83
              '../data//Yoga_TEST',  # 84
              '../data//ACSF1_TEST',  # 85
              '../data//AllGestureWiimoteX_TEST',  # 86
              '../data//AllGestureWiimoteY_TEST',  # 87
              '../data//AllGestureWiimoteZ_TEST',  # 88
              '../data//BME_TEST',  # 89
              '../data//Chinatown_TEST',  # 90
              '../data//Crop_TEST',  # 91
              '../data//DodgerLoopDay_TEST',  # 92
              '../data//DodgerLoopGame_TEST',  # 93
              '../data//DodgerLoopWeekend_TEST',  # 94
              '../data//EOGHorizontalSignal_TEST',  # 95
              '../data//EOGVerticalSignal_TEST',  # 96
              '../data//EthanolLevel_TEST',  # 97
              '../data//FreezerRegularTrain_TEST',  # 98
              '../data//FreezerSmallTrain_TEST',  # 99
              '../data//Fungi_TEST',  # 100
              '../data//GestureMidAirD1_TEST',  # 101
              '../data//GestureMidAirD2_TEST',  # 102
              '../data//GestureMidAirD3_TEST',  # 103
              '../data//GesturePebbleZ1_TEST',  # 104
              '../data//GesturePebbleZ2_TEST',  # 105
              '../data//GunPointAgeSpan_TEST',  # 106
              '../data//GunPointMaleVersusFemale_TEST',  # 107
              '../data//GunPointOldVersusYoung_TEST',  # 108
              '../data//HouseTwenty_TEST',  # 109
              '../data//InsectEPGRegularTrain_TEST',  # 110
              '../data//InsectEPGSmallTrain_TEST',  # 111
              '../data//MelbournePedestrian_TEST',  # 112
              '../data//MixedShapesRegularTrain_TEST',  # 113
              '../data//MixedShapesSmallTrain_TEST',  # 114
              '../data//PickupGestureWiimoteZ_TEST',  # 115
              '../data//PigAirwayPressure_TEST',  # 116
              '../data//PigArtPressure_TEST',  # 117
              '../data//PigCVP_TEST',  # 118
              '../data//PLAID_TEST',  # 119
              '../data//PowerCons_TEST',  # 120
              '../data//Rock_TEST',  # 121
              '../data//SemgHandGenderCh2_TEST',  # 122
              '../data//SemgHandMovementCh2_TEST',  # 123
              '../data//SemgHandSubjectCh2_TEST',  # 124
              '../data//ShakeGestureWiimoteZ_TEST',  # 125
              '../data//SmoothSubspace_TEST',  # 126
              '../data//UMD_TEST',  # 127
              '../data//Synthetic_TEST.txt', #128
              '../data//Depresjon_TEST.txt'
              ]

