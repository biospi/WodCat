import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

"""
Utility class for static methods
"""


def random_permutations(input_list, k):
    result = []
    # Fill the reservoir with the first k elements
    for i in range(len(input_list)):
        if i < k:
            result.append(input_list[i])
        else:
            # Randomly choose an index to potentially replace
            j = random.randint(0, i)
            if j < k:
                result[j] = input_list[i]
    return result


def center_signal(y, avg):
    y_centered = y - avg
    return y_centered


def create_rec_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def getXY(df):
    print(df)
    X = df.iloc[:, :-2].values
    y = df["health"].values
    return X, y


def binarize(tagets, healty_target=1):
    return (tagets != healty_target).astype(int)


def concatenate_images(images_list, out_dir, filename="cwt_mean_per_label.png"):
    imgs = [Image.open(str(i)) for i in images_list]

    # If you're using an older version of Pillow, you might have to use .size[0] instead of .width
    # and later on, .size[1] instead of .height
    min_img_width = min(i.width for i in imgs)

    total_height = 0
    for i, img in enumerate(imgs):
        # If the image is larger than the minimum width, resize it
        if img.width > min_img_width:
            imgs[i] = img.resize((min_img_width, int(img.height / img.width * min_img_width)), Image.ANTIALIAS)
        total_height += imgs[i].height

    # I have picked the mode of the first image to be generic. You may have other ideas
    # Now that we know the total height of all of the resized images, we know the height of our final image
    img_merge = Image.new(imgs[0].mode, (min_img_width, total_height))
    y = 0
    for img in imgs:
        img_merge.paste(img, (0, y))

        y += img.height

    file_path = out_dir.parent / filename
    print(file_path)
    img_merge.save(str(file_path))


def ninefive_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]


def explode(df, columns):
    df['tmp'] = df.apply(lambda row: list(zip(row[columns])), axis=1)
    df = df.explode('tmp')
    df[columns] = pd.DataFrame(df['tmp'].tolist(), index=df.index)
    df.drop(columns='tmp', inplace=True)
    print(df)
    return df


def purge_hpc_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def create_batch_script(uob_username, account, commands, num_commands):
    file_content = '''#!/bin/env bash
#SBATCH --account={account}
#SBATCH --job-name=cats_paper
#SBATCH --output=cats_paper
#SBATCH --error=cats_paper
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH --mem=100000M
#SBATCH --array=1-{num_commands}

# Load the modules/environment
module purge
module load languages/anaconda3/3.7
conda init
source ~/.bashrc


# Define working directory
export WORK_DIR=/user/work/{uob_username}/WodCat

# Change into working directory
cd ${{WORK_DIR}}
source /user/work/{uob_username}/WodCat/venv/bin/activate

# Do some stuff
echo JOB ID: ${{SLURM_JOBID}}
echo PBS ARRAY ID: ${{SLURM_ARRAY_TASK_ID}}
echo Working Directory: $(pwd)

cmds=({commands})
# Execute code
echo ${{cmds[${{SLURM_ARRAY_TASK_ID}}]}}
python ${{cmds[${{SLURM_ARRAY_TASK_ID}}]}} > /user/work/{uob_username}/logs/cat_paper_${{SLURM_ARRAY_TASK_ID}}.log
'''
    # Format the content with provided commands and number of commands
    formatted_content = file_content.format(
        uob_username=uob_username,
        account=account,
        commands=' '.join(f"{cmd}" for cmd in commands),
        num_commands=num_commands
    )

    filepath = Path(os.path.abspath(__file__)).parent / "bc4_cats_cpu.sh"
    print(f"File ready at: {filepath}")
    # Write the content to the file
    with open(filepath, "w") as file:
        file.write(formatted_content)


def time_of_day(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return 'Noon'
    elif (x > 16) and (x <= 20):
        return 'Eve'
    elif (x > 20) or (x <= 4):  # Merge Night and Late Night
        return 'Night/Late Night'


if __name__ == "__main__":
    # Example usage:
    command_list = ['command1', 'command2', 'command3']
    num_of_commands = len(command_list)
    create_batch_script('fo18103', 'sscm012844', command_list, num_of_commands)

    # # Example usage with input list [1, 2, 3, ..., 1000000] and k = 10
    # input_list = list(range(1, 1000001))
    # k = 10
    # selected_permutations = [random_permutations(input_list, 3) for _ in range(k)]
    #
    # print(selected_permutations)