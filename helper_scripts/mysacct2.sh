# just to check when I have submitted. I need to set time
sacct -S 08/10-19:00 --format JobID,State,Submit,Start | sed -n '1~2p' 
