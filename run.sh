###
 # @Author: Mengjie Zheng
 # @Email: mengjie.zheng@colorado.edu;zhengmengjie18@mails.ucas.ac.cn
 # @Date: 2023-07-11 18:35:25
 # @LastEditors: Mengjie Zheng
 # @LastEditTime: 2024-06-21 09:38:54
 # @FilePath: /Projects/Alaska.Proj/MCMC_Compliance/run.sh
 # @Description: 
### 


workdir=/Users/mengjie/Projects/Alaska.Proj/MCMC_Compliance
datadir=/Volumes/SeisBig23/mengjie_data/Alaska.Data/COMPLY_INV
logdir=$datadir/LOG/Inversion
runlst=$workdir/run.lst

date=$(date +"%Y%m%d")

cat $runlst | parallel -j 10 "echo 'Running: {}'; eval '{}; (time {})' &> $logdir/\`echo {} | awk -F' ' '{print \$NF}'\`.$date.INV.log"

# cat $runlst | parallel -j 6 "echo 'Running: {}'; (time eval '{}') &> $logdir/\`echo {} | awk -F' ' '{print \$NF}'\`.$date.INV.log"

# cat $runlst | parallel -j 8 \
#   "echo 'Running: {}'; \
#   (time eval '{}') 2>&1 | \
#   tee $logdir/\`echo {} | awk -F' ' '{print \$NF}'\`.$date.INV.log; \
#   exit_status=\$?; \
#   if [ \$exit_status -ne 0 ]; then \
#     echo 'WARNING: Command failed with exit status \$exit_status' >&2; \
#   fi"