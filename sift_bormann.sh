for imgname in labc intel labf laba labd f101 #
do 
    mkdir ${imgname} && cd ${imgname}

    if [ "${imgname}" == labc ]
    then
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_c/lab_c_scan.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_c/lab_c_scan_lab_c_15.png' 
    elif [ "${imgname}" == intel ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/intel/intel_clean.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/intel/intel0_54.png' 
    elif [ "${imgname}" == labf ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_f/lab_e0.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_f/lab_f_scan_1270.png' 
    elif [ "${imgname}" == laba ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_a/lab_a.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_a/lab_a_lab_a_scan_-112.png' 
    elif [ "${imgname}" == labd ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_d/lab_d0.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/lab_d/lab_d0_lab_d_scan_furnitures_31_.png' 
    elif [ "${imgname}" == f101 ] 
    then 
        python ../sift_FB.py --img_src '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/freiburg_building101/freiburg_building101.png' --img_dst '/home/houjw/Seafile/Seafile/dataset_results/dataset_matching/bormann/freiburg_building101/freiburg_building101_Freiburg101_scan_-145_.png' 
    else
        echo "unknow parameter!!!!!!"
    fi
    cd ..

done 
        
