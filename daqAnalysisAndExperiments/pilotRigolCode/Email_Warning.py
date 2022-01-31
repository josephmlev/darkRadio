#This code will send a warning email about dark radio shutdowns
#Daniel Polin 2019

import sys, smtplib
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart

def Send_Warning(message_subject, message_text):
    to_list=['dapolin@ucdavis.edu','bpgodfrey@ucdavis.edu','tyson@physics.ucdavis.edu','tyson.physics@gmail.com','hillbrand@ucdavis.edu','jmlev@ucdavis.edu','sklomp@ucdavis.edu']
    msg = MIMEMultipart()
    msg['From']='darkradioerrors@gmail.com'
    msg['Subject']=message_subject
    msg.attach(MIMEText(message_text,'plain'))
    server=smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('darkradioerrors@gmail.com','sourcescanvertical')
    for to_addr in to_list:
        msg['To']=to_addr
        text=msg.as_string()
        server.sendmail('darkradioerrors@gmail.com', to_addr, text)
    server.quit()
    return 

Send_Warning("Warning: Dark Radio Shutdown", "The dark radio script has shut down unexpectedly.")

