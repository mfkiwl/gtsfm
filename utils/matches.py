

def convert_match_indices_to_coordinates(feature_coords_1,
                                         feature_coords_2,
                                         match_indices):
    # TODO: most probably unused
    return feature_coords_1[match_indices[:, 0], :], feature_coords_2[match_indices[:, 1], :]
