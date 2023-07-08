
import cv2
import numpy as np
import re



messages = ["\nPROCESSING IMAGE #", "[READING]", "[DETECTING]", "[SAVING]", "[COMPLETED WITH IoU SCORE]: "]
divider = "	--------------------	"



def main():

	weights = ["c", "d", "i", "j", "y", "z"]
	models = model_reader(weights)
	
	for i in range(1,31):
		
		input_code = str(i)
		
		image_path, input_code = path_reader(input_code)
		print(divider + messages[0] + input_code + divider)
		
		print(messages[1])
		img = image_reader(image_path)
		og = img.copy()
		
		print(messages[2])
		detections, nms_indexes = detector(img, models)
		
		print(messages[3])
		img, good_boxes = output_creator(img, detections, nms_indexes, weights, input_code)
		
		
		score = IoU_calculation(og, good_boxes, input_code)
		print(messages[4] + str(score)[0:5])
		
		cv2.imshow("Detections for image #" + input_code, img)
		key = cv2.waitKey(0)
		
	cv2.destroyAllWindows()






def model_reader(weights):
	models = []
	for x in weights: 
		net = cv2.dnn.readNet("weights/"+x+".weights", "cfg/yolov3_testing.cfg")
		layer_names = net.getLayerNames()
		out_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		models.append([net, layer_names, out_layers])
	
	return models



		

def path_reader(input_code):
	
	if (int(input_code) < 10):
		input_code = "0" + input_code
			
	img_path = "in_imgs/" + input_code + ".jpg"
		
	return img_path, input_code





def image_reader(image_path):
	return cv2.imread(image_path)




	
def detector(img, models):

	detections = []
	
	height, width, channels = img.shape
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) 

	for i in range(len(models)):
		model = models[i]
		net = model[0]
		layer_names = model[1]
		out_layers = model[2]

		net.setInput(blob)
		outs = net.forward(out_layers)

		for out in outs:
				for detection in out:
					scores = detection[5:]
					class_id = np.argmax(scores)
					confidence = scores[class_id]
					if confidence > 0.2:
						# Object detected
						center_x = int(detection[0] * width)
						center_y = int(detection[1] * height)
						w = int(detection[2] * width)
						h = int(detection[3] * height)

						# Rectangle coordinates
						x = int(center_x - w / 2)
						y = int(center_y - h / 2)
						
						detections.append([[x, y, w, h], float(confidence), i])
						
	nms_indexes = cv2.dnn.NMSBoxes([d[0] for d in detections], [d[1] for d in detections], 0.35, 0.2)
	return detections, nms_indexes





def output_creator(img, detections, nms_indexes, weights, input_code):
	colors = np.random.uniform(0, 255, size=(len(weights), 3))
	
	boxes = [d[0] for d in detections] 
	confidences = [d[1] for d in detections] 
	net_ids = [d[2] for d in detections] 

	good_boxes =  []
	for i in range(len(detections)):
		if i in nms_indexes:
			good_boxes.append(boxes[i])
			x, y, w, h = boxes[i]
			label = str(weights[net_ids[i]]) + ":" + str(confidences[i])[0:3]
			cv2.rectangle(img, (x, y), (x + w, y + h), colors[net_ids[i]], 2)
			#cv2.putText(img, label, (x, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
	    
	bl = " "        
	with open("out_boxes/" + input_code + ".txt", 'w') as f:
		for i in range (len(good_boxes)):
			x, y, w, h = good_boxes[i]
			f.write(str(x) + bl + str(y) + bl + str(w) + bl + str(h))
			f.write('\n')

	cv2.imwrite("out_imgs/" + input_code + ".jpg", img)
	print("Saved: " + input_code + ".jpg & " + input_code + ".txt")
	
	return img, good_boxes
	
	
	
	
def IoU_calculation(img, good_boxes, input_code):
	height, width, chn = img.shape
	det_copy = np.zeros((height, width, 1), dtype = "uint8")
	gt_copy = np.zeros((height, width, 1), dtype = "uint8")
			
	for box in good_boxes:
		x, y, w, h = box
		for i in range(x, x+w):
			for j in range (y, y+h):
				if ((i<width) and (j<height)):
					det_copy[j][i] = 255
	
	lines = []
	with open("GT_boxes/" + input_code + ".txt") as f:
		lines = f.readlines()
    	
	for line in lines:
		x, y, w, h = map(int, re.findall(r'\d+', line))
		for i in range (x, x+w):
			for j in range (y, y+h):
				if ((i<width) and (j<height)):
					gt_copy[j][i] = 255
		
	intersection = 0
	union = 1
	
	for i in range(len(gt_copy)):
		for j in range(len(gt_copy[i])):
			if (gt_copy[i][j] == 255 or det_copy[i][j] == 255):
				union+=1
				
			if (gt_copy[i][j] == 255 and det_copy[i][j] == 255):
				intersection+=1	
				
	score = intersection/union
	return score


main()



