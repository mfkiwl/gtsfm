/* Component to render Frontend Summary metrics. Specifically rotation sucesses, translation
successes, pose successes, and correspondence inliers.

Author: Adi Singh
*/
import React from "react";

// Local Imports.
import '../stylesheets/FrontendSummary.css'

function FrontendSummary(props) {
    /*
    Args:
        props.json.rotation.success_count (int): Number of image pairs with estimated relative rotation error 
        underneath the threshold.
        props.json.translation.success_count (int): Number of image pairs with estimated relative translation error 
        underneath the threshold.
        props.json.num_valid_entries (int): Number of valid image pair entries.
        props.json.num_total_entries (int): Number of total image pair entries.
        props.json.pose.success_count (int): Number of image pairs with successfully recovered relative pose.
        props.json.correspondence.all_inliers (int): Number of correspondence inliers.
        
    Returns:
        A component showing frontend summary metrics after clicking the 'TwoViewEstimator' Plate.
    */

    return (
        <div className="fs_container">
            <div className="fs_sub_container">
                <h3>Frontend Summary Metrics</h3>
                <p className="format_hint">
                    Format: (success count)/(valid entries)/(total entries)
                </p>

                <p className="fs_text">
                    Rotation Success: {props.json.rotation.success_count}/
                                    {props.json.num_valid_entries}/
                                    {props.json.num_total_entries}
                </p>

                <p className="fs_text">
                    Translation Success: {props.json.translation.success_count}/
                                        {props.json.num_valid_entries}/
                                        {props.json.num_total_entries}
                </p>

                <p className="fs_text">
                    Pose Success: {props.json.pose.success_count}/
                                {props.json.num_valid_entries}/
                                {props.json.num_total_entries}
                </p>

                <p className="fs_text">
                    Correspondence Inliners: {props.json.correspondences.all_inliers}
                </p>

                <button className="go_back_btn" onClick={() => props.toggleFS(false)}>
                    Go Back
                </button>
            </div>
        </div>
    )
}

export default FrontendSummary;