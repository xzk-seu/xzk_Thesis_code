labels = ['<PAD>', 'O', 'B-Data_Structure', 'B-Code_Block', 'I-Code_Block', 'B-Application', 'I-Application', 'B-Library_Function', 'B-Function_Name', 'B-Data_Type', 'B-Language', 'B-Library', 'B-Variable_Name', 'B-Value', 'I-Value', 'B-Device', 'B-User_Name', 'B-User_Interface_Element', 'B-Output_Block', 'I-Output_Block', 'B-Error_Name', 'B-Class_Name', 'B-Library_Class', 'I-Function_Name', 'B-Website', 'I-Library', 'I-Library_Function', 'B-Version', 'I-Library_Class', 'B-File_Name', 'I-User_Name', 'I-Data_Structure', 'B-File_Type', 'I-User_Interface_Element', 'B-Library_Variable', 'B-Operating_System', 'I-Device', 'I-Data_Type', 'B-Algorithm', 'B-Organization', 'I-Algorithm', 'I-Version', 'B-HTML_XML_Tag', 'I-Error_Name', 'I-File_Name', 'I-Variable_Name', 'I-Class_Name', 'I-Operating_System', 'I-Website', 'I-Organization', 'I-File_Type', 'I-HTML_XML_Tag', 'I-Library_Variable', 'B-Keyboard_IP', 'B-Licence', 'I-Licence', 'I-Language', 'I-Keyboard_IP', '<START>', '<STOP>']


c = 0
for i in labels:
    if i[0] == "B":
        c += 1
        print(i.split("-")[-1])

