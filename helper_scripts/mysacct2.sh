# just to check when I have submitted. I need to set time
sacct -S 11/10-00:00 --format JobID,State,Submit,Start | sed -n '1~2p' 
