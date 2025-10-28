import os
#from skimage.metrics import structural_similarity as ssim
 
inclusion_list = [1,2,3,4]
categories_list = [1,2,3,4,5,6,7]
 
folder_ground_truth = "GroundTruths"
folder_training_data = "TrainingDataNan"
#folder_reconstruction = "Output"
#folder_output_examples = "gbar/results"
 
def submit(jobid,cmd):
    id = str(jobid)
    jobname = 'job_' + id
    memcore = 7000
    maxmem = 8000
    email = 'amaal@dtu.dk'
    ncores = 1
 
    # begin str for jobscript
    strcmd = '#!/bin/sh\n'
    strcmd += '#BSUB -J ' + jobname + '\n'
    strcmd += '#BSUB -q hpc\n'
    strcmd += '#BSUB -n ' + str(ncores) + '\n'
    strcmd += '#BSUB -R "span[hosts=1]"\n'
    strcmd += '#BSUB -R "rusage[mem=' + str(memcore) + 'MB]"\n'
    strcmd += '#BSUB -M ' + str(maxmem) + 'MB\n'
    strcmd += '#BSUB -W 10:00\n'
    strcmd += '#BSUB -u ' + email + '\n'
    strcmd += '#BSUB -N \n'
    strcmd += '#BSUB -o gbar/output/output_' + id + '.out\n'
    strcmd += '#BSUB -e gbar/error/error_' + id + '.err\n'
    strcmd += 'module load FEniCS/2019.1.0-with-petsc-3.15.5-numpy-1.23.3\n'
    strcmd += 'source fenics2019/bin/activate\n'
    strcmd += cmd
 
    jobscript = 'gbar/submit_'+ jobname + '.sh'
    f = open(jobscript, 'w')
    f.write(strcmd)
    f.close()
    os.system('bsub < ' + jobscript)
 
if __name__ == "__main__":
    TV_factor=5e5
    Tikhonov_factor = 0.5
    CUQI1_factor = 1e10
    num_iter = 150
    segmentation_method = 'otsu'

    for category in [1,2,3,4,5,6,7]:
        # before new all iterations are 70
        # new2, fix otsu name
        tag = 'new2_'+str(TV_factor)+'_'+str(Tikhonov_factor)+'_'+str(CUQI1_factor)+'_'+str(num_iter)+'_'+str(category) + '_' + segmentation_method
        outputFolder = 'gbar/output'+tag
        #folder_reconstruction = 'gbar/' + str(category)
        cmd = 'python main_2.py' + ' ' + folder_training_data+str(category) + ' ' + outputFolder + ' ' + str(category) + ' ' + str(num_iter) + ' ' + str(TV_factor) + ' ' + str(Tikhonov_factor) + ' ' + str(CUQI1_factor) + ' ' + segmentation_method
        print(cmd)
        submit(tag,cmd)

 