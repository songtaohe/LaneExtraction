for tag in 0 5 6 11 12 17 18 22 25 28 31
do
    time python evaluate_v0_lane_extraction_only.py ../dataset_evaluation/ _${tag}
done