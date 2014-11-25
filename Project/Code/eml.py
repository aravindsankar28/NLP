import email,sys
file_name = sys.argv[1]
j= sys.argv[2]

with open(file_name) as f:
	raw_message = f.read()

msg = email.message_from_string(raw_message)


sub = open('Subjects/Subject_'+j+'.txt','w')

mail = open('Mails/Mail_'+j+'.txt','w')

sub.write(msg['Subject']+"\n")
sub.write(msg['from'])
for part in msg.walk():
    # each part is a either non-multipart, or another multipart message
    # that contains further parts... Message is organized like a tree
    if part.get_content_type() == 'text/plain':
        mail.write(part.get_payload()) # prints the raw te


sub.close()
mail.close()