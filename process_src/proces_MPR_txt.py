import os
import numpy as np

def get_test_run_dirnames (roller_dir: str):
    
    """
    Reads in roller_dir the radiusloss text files to find the good directory names with data
    
    Args:
        roller_dir (str): Directory with the _RadiusLoss.txt files
        
    Returns:
        dirnames (list): list of data directory names
        
    """
    dirnames = []
    
    for file in os.listdir(roller_dir):
        
        if file.endswith('_RadiusLoss.txt'):
            
            dirnames += [file.removesuffix('_RadiusLoss.txt')]
            
    return dirnames


def remove_duplicate_cols (all_data: np.ndarray, column_names: list):
    
    """
    Gets rid of duplicate columns with the specified data/column_names pair
    
    Args:
        all_data (np.ndarray): array of concatented data for 1 test run
        column_names (list): list of column names
        
    Returns:
        no_dup_data, col_names
        
        no_dup_data (np.ndararay): array with duplicates removed
        col_names (list): corresponding columns
    """
    unique_cols = []
    duplicat_idx = []
    
    for idx, current_col_name in enumerate(column_names):
        
        if unique_cols.count(current_col_name) ==0:
            
            unique_cols += [current_col_name]
        else:
            duplicat_idx += [idx]

    new_data = np.delete(all_data, duplicat_idx, axis = 1)

    return new_data, unique_cols
            
        
    
    



def get_column_names (run_folder: str):
    
    """
    Returns the column names of columns that have non-empty data to symplify populating a dataframe
    
    Args:
        run_folder (str): folder with ll stages of 1 test: they all need to have the same columns
        
    Returns:
        col_names
        
        col_names (list): list of all variable names in same order as the stage files
    """
    
    sorted_dirlist = [file for file in os.listdir(run_folder) if file.endswith('.txt')]
    _order_dir(sorted_dirlist)
    
    filepath = '/'.join([run_folder, sorted_dirlist[0]])
    
    with open(filepath, 'r') as f:
        
        found_names = False
        
        while not found_names:
            
            line = next(f).split()
            # checking the len(line) > 0 to skip empty lines
            if len(line) > 0 and line[0] == 'Step' and line[1] == "1":
                
                # Go two lines after start of step to find column names
                line = next(f)
                line = next(f).split(sep = '	')
                
                
                # Get only the names of used data columns
                all_col_names = line [:-1]
                data_line = next(f).split(sep = '	')
                
                col_names = [all_col_names[i] for i in range(len(all_col_names)) if len(data_line[i])>0]
            
                found_names = True
                
    
    return col_names
    
    
    
    
        


def get_testrun_data (run_folder : str):
    
    """
    Extract the test run dta from the specified folder, while keeping track of indices of steps and stages
    
    Args:
        run_folder (str): full path of the folder with all run stages
        
    Returns:
        run_data , step_indices 
         
        run_data (np.ndarray): the full data contained in all stages combined
        step_indices (list): a list of lists to keep track of step and stage indices. Example: with two files, so two tages, step_indices would look like: 
            [
                [0, k1, n], 
                [n, k2, k3, m]
            ]
        
    """
    
    run_data = []
    step_indices = []
    
    sorted_dirlist = [file for file in os.listdir(run_folder) if file.endswith('.txt')]
    # sorts name of files such that we process steps from 1 to k in chronological order
    _order_dir(sorted_dirlist)
    
    
    for filename in sorted_dirlist:
        
        filepath = '/'.join([run_folder, filename])
        
        if len(step_indices) == 0:
            
            # Initilize step indices with [0] only for the first run
            file_data, file_indices = _get_stagetxt_data(filepath, [0])
            
            run_data += [file_data]
            step_indices += [file_indices]
        
        else:
            
            
            file_data, file_indices = _get_stagetxt_data(filepath, step_indices[-1])
            
            run_data += [file_data]
            step_indices += [file_indices]
    
    
    return np.concatenate(run_data, axis = 0), step_indices





def _get_stagetxt_data(file_name: str, prev_steps:list):
    
    """
    Extracts the data from the specified text stage file 
    
    Args:
        file_name (str): full path of the text stage file
        prev_steps (list): Last step indices of the current test run we are etracting data from. Mut be 1D
    
    Returns:
        file_data (np.ndarray), new_steps (list)
        file_data : numpy array with every column and every data line
        new_steps: list of the bounding indices (for example, first file with 2 steps accepts [0] and will return [0, k+1, n+1] with n lenght of file and k index of end of first step)
    """
    
    curr_glo_index = prev_steps[-1]
    file_data = []
    new_steps = [prev_steps[-1]]
    
    with open(file_name, 'r') as f:
        
        end = False
        
        # All stages start with a step index of 1
        step_index = 1
        
        # We are sure the first line as at lest 1 chracter so .split() will not throw an error
        line = next(f).split()
        
        while not end:
            
            # checking the len(line) > 0 to skip empty lines
            if len(line) > 0 and line[0] == 'Step' and line[1] == f"{step_index}":
                
                # Need to skip the next two lines
                line = next(f)
                line = next(f)
                
                # instantiate first data_line of current step
                line = next(f).split()
                
                while len(line) > 0:
                    
                    # Check if number and if yes add it to line_data
                    line_data = []
                    for element in line:
                        try:
                            element = float(element)
                            line_data += [element]
                        except ValueError:
                            pass
                        
                    # finish current line, go to next one and increase global index
                    line = next(f).split()
                    curr_glo_index += 1
                
                    # update 
                    file_data += [line_data]
                
                new_steps += [curr_glo_index]
                step_index += 1
            
            
            
            # At end of loop, check if next line exist or not
            line = next(f, None)
            if line == None:
                end = True
            else:
                # instantiate next line i it exists
                line = line.split()
    
    
    return np.array(file_data), new_steps
    
    
    
def _order_dir(dirlist:list):
    
    """
    In-place sorting of an array for text files (.txt) that have their last character as a number
    
    Args:
        dirlist (list): list with all file_names of a directory
    """
    
    for start_i in range(len(dirlist)):
        
        # the [-5] element of the name of a .txt file is the last character of the actual name
        min_stage = int(list(dirlist[start_i])[-5])
        idx_min = start_i
        
        for j in range (start_i + 1, len(dirlist)):
            
            if int(list(dirlist[j])[-5]) < min_stage:
                min_stage = int(list(dirlist[j])[-5])
                idx_min = j
                
            
        dirlist[start_i], dirlist[idx_min] = dirlist[idx_min], dirlist[start_i]

    