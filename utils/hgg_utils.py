

import pathlib

def get_hgg_paths():
    
    """
    This will only work if imported in a file that lives inside the S20-team0-project directory
    Abbreviated expected directory setup for path to work:

    ./some_directory_to_hold_everything/
        +-- MICCAI_BraTS_2019_Data_Training
        |
        |   +-- MICCAI_BraTS_2019_Data_Training
        |
        |   |   + -- HGG
        |
        +-- S20-team0-project  */ Cloned github repo */
        
    Return path to HGG directory
    """
    
    hgg_folders_path = sorted(
                            list(
                                sorted(
                                    list(
                                        sorted ( 
                                                list( 
                                                        pathlib.Path.cwd().parent.iterdir() 
                                                ) 
                                        )[7].iterdir()
                                    )
                                )[0].iterdir()
                            )
                        )[0]

    return hgg_folders_path


def get_each_hgg_folder( ):
    
    """
    This will only work if imported in a file that lives inside the S20-team0-project directory
    Abbreviated expected directory setup for path to work:

    Returns a sorted list of paths to each HGG folder
    """
    
    return sorted( [folder for folder in get_hgg_paths().iterdir() ] )
    
def get_scans_at_index( i ):
    
    """
    This will only work if imported in a file that lives inside the S20-team0-project directory
    Abbreviated expected directory setup for path to work:

    Returns a sorted list of the 5 modalities inside hgg folder at index i
    """
    
    return sorted( [modality for modality in get_each_hgg_folder()[i].iterdir()] )
    
    