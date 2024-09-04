import pathlib  # For working with file paths

from fpdf import FPDF  # For creating PDF documents


class ReportPDF():
    '''
    This class intended to create a PDF report including the images generated during the pipeline.
    
    Parameters
    
    directory_results: str or PosixPath
        Directory containing the images to include in the report.
    channels_with_FISH  : list of int
        List with integers indicating the index of channels for the FISH detection using.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the FISH plot detection. The default is True.
    list_z_slices_per_image : int
        List containing all z-slices for each figure.
        
    .. image:: images/pdf_report.png
    
    This PDF file is generated, and it contains the processing steps for each image in the folder.
    
    '''    
    def __init__(self, directory,filenames_for_pdf_report, channels_with_FISH,save_all_images,list_z_slices_per_image,threshold_for_spot_detection,list_segmentation_successful=True):
        self.directory = directory
        if isinstance(channels_with_FISH, list): 
            self.channels_with_FISH = channels_with_FISH
        else:
            self.channels_with_FISH = [channels_with_FISH]
        self.save_all_images = save_all_images
        self.list_z_slices_per_image = list_z_slices_per_image
        self.threshold_for_spot_detection=threshold_for_spot_detection
        self.list_segmentation_successful =list_segmentation_successful
        self.filenames_for_pdf_report=filenames_for_pdf_report
        
    def create_report(self):
        '''
        This method creates a PDF with the original images, images for cell segmentation and images for the spot detection.
        '''
        pdf = FPDF()
        WIDTH = 210
        HEIGHT = 297
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        # code that reads the main file names
        list_files_names = self.filenames_for_pdf_report #[]
        # Main loop that reads each image and makes the pdf
        for i,temp_file_name in enumerate(list_files_names):
            pdf.cell(w=0, h=10, txt='Original image: ' + temp_file_name,ln =2,align = 'L')
            # code that returns the path of the original image
            temp_original_img_name = pathlib.Path().absolute().joinpath( self.directory, 'ori_' + temp_file_name +'.png' )
            pdf.image(str(temp_original_img_name), x=0, y=20, w=WIDTH-30)
            # creating some space
            for text_idx in range(0, 12):
                pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
            pdf.cell(w=0, h=10, txt='Cell segmentation: ' + temp_file_name,ln =1,align = 'L')
            # code that returns the path of the segmented image
            if self.list_segmentation_successful[i]==True:
                temp_segmented_img_name = pathlib.Path().absolute().joinpath( self.directory, 'seg_' + temp_file_name +'.png' )
                pdf.image(str(temp_segmented_img_name), x=0, y=HEIGHT/2, w=WIDTH-30)
            else:
                pdf.cell(w=0, h=20, txt='Segmentation was not possible for image: ' + temp_file_name,ln =1,align = 'L')
                pdf.add_page()
            # Code that plots the detected spots.
            if (self.save_all_images==True) and (self.list_segmentation_successful[i]==True):
                for id_channel, channel in enumerate(self.channels_with_FISH):
                    counter=1
                    pdf.add_page() # adding a page
                    for z_slice in range(0, self.list_z_slices_per_image[i]):
                        temp_seg_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '_ch_'+str(channel) + '_slice_'+ str(z_slice) +'.png' )
                        # Plotting bottom image
                        if counter%2==0: # Adding space if is an even counter
                            # adding some space to plot the bottom image
                            for j in range(0, 11):
                                pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
                            # Plotting the image
                            try:
                                pdf.cell(w=0, h=0, txt='FISH Ch_ ' + str(channel) + '_slice_'+ str(z_slice) +': '+ temp_file_name,ln =2,align = 'L')
                                pdf.image(str(temp_seg_name), x=0, y=HEIGHT//2, w=WIDTH-80)
                                pdf.add_page()
                            except:
                                pass
                        # plotting top image
                        else:
                            try:
                                pdf.cell(w=0, h=10, txt='FISH Ch_ ' + str(channel) + '_slice_'+ str(z_slice) +': '+ temp_file_name,ln =2,align = 'L')
                                pdf.image(str(temp_seg_name), x=0, y=20, w=WIDTH-80)
                            except:
                                pass
                        counter=counter+1
                    pdf.add_page()
                    try:
                        if (self.threshold_for_spot_detection[id_channel] is None):
                            temp_elbow_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '__elbow_'+ '_ch_'+str(channel)+'.png' )
                            pdf.image(str(temp_elbow_name), x=0, y=HEIGHT//2, w=WIDTH-140)
                        else:
                            pdf.cell(w=0, h=10, txt='Used intensity threshold = '+str(self.threshold_for_spot_detection[id_channel]) ,ln =2,align = 'L')
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')
                    pdf.add_page()
            elif self.list_segmentation_successful[i]==True:
                pdf.add_page()
                for id_channel, channel in enumerate(self.channels_with_FISH):
                    # Plotting the image with detected spots
                    temp_seg_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '_ch_'+str(channel)+'.png' )
                    pdf.cell(w=0, h=10, txt='FISH Ch_ ' + str(channel) + ': '+ temp_file_name,ln =2,align = 'L') 
                    try:
                        pdf.image(str(temp_seg_name), x=0, y=20, w=WIDTH-30)  
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')                  
                    # adding some space
                    for j in range(0, 12):
                        pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
                    # Plotting the elbow plot
                    try:
                        if (self.threshold_for_spot_detection[id_channel] is None):
                            temp_elbow_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '__elbow_'+ '_ch_'+str(channel)+'.png' )
                            pdf.image(str(temp_elbow_name), x=0, y=HEIGHT//2, w=WIDTH-140)
                        else:
                            pdf.cell(w=0, h=10, txt='Used intensity threshold = '+str(self.threshold_for_spot_detection[id_channel]) ,ln =2,align = 'L')
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')
                    pdf.add_page()                
        pdf_name =  'pdf_report_' + self.directory.name[13:].replace(" ", "") + '.pdf'
        pdf.output(pdf_name, 'F')
        return None
    