for imgname in intel LongCorridor LongRun robot_cast_0_1 robot_cast_0_2 robot_cast_0_3 robot_cast_1_2 robot_cast_1_3 robot_cast_2_3 
do 
    mkdir ${imgname} && cd ${imgname}

    if [ "${imgname}" == intel ]
    then
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/intel/intel.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/intel/intel_105.png' 
    elif [ "${imgname}" == LongCorridor ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/LongCorridor/LongCorridor2_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/LongCorridor/LongCorridor10.png' 
    elif [ "${imgname}" == LongRun ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/LongRun/LongRun6_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/LongRun/LongRun_115.png' 
    elif [ "${imgname}" == robot_cast_0_1 ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_0_1/robot0_cast_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_0_1/robot1_cast_0_1_61_05.png' 
    elif [ "${imgname}" == robot_cast_0_2 ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_0_2/robot0_cast_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_0_2/robot2_cast_0_2_-45_71.png' 
    elif [ "${imgname}" == robot_cast_0_3 ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_0_3/robot0_cast_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_0_3/robot3_cast_0_3_66_66.png' 
    elif [ "${imgname}" == robot_cast_1_2 ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_1_2/robot1_cast_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_1_2/robot2_cast_2463.png' 
    elif [ "${imgname}" == robot_cast_1_3 ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_1_3/robot1_cast_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_1_3/robot3_cast_4131.png' 
    elif [ "${imgname}" == robot_cast_2_3 ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_2_3/robot2_cast_.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/img/robot_cast_2_3/robot3_cast_2_3_17.png' 
    else
        echo "unknow parameter!!!!!!"
    fi
    cd ..
done