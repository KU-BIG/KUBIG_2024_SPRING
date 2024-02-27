import cv2
import numpy as np
import torch
from pathlib import Path

# =============== basic config =================
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
"LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Nose", "Neck"], ["Nose", "REye"], ["Nose", "LEye"], ["Neck", "RShoulder"], 
              ["Neck", "LShoulder"], ["Neck", "RHip"], ["Neck", "LHip"], ["RShoulder", "RElbow"], 
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["RHip", "RKnee"], 
              ["RKnee", "RAnkle"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["REye", "REar"], ["LEye", "LEar"]]

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                    [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

protoFile_coco = "./models/coco/pose_deploy_linevec.prototxt"
weightsFile_coco = "./models/coco/pose_iter_440000.caffemodel"
gt_img_path = "/Users/gimhyeon-u/Documents/Eunwoooo/kubig/project1/yoga82/train/Warrior_II_Pose_or_Virabhadrasana_II_/Warrior_II_Pose_or_Virabhadrasana_II__image_19.jpg"

# =============== network model =================
net = cv2.dnn.readNetFromCaffe(protoFile_coco, weightsFile_coco)
inputWidth=320
inputHeight=240
inputScale=1.0/255

# =============== keypoint extraction =================
def output_keypoints_with_lines_image(img):
    inpBlob = cv2.dnn.blobFromImage(img, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    imgb=cv2.dnn.imagesFromBlob(inpBlob)
    net.setInput(inpBlob)
    output = net.forward()
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    points = []
    for i in range(0,19):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (imgWidth * point[0]) / output.shape[3]
        y = (imgHeight * point[1]) / output.shape[2]
 
        if prob > 0.1 :    
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED) # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)
    
    for pair in POSE_PAIRS:
        partA = pair[0]             
        partA = BODY_PARTS[partA]   
        partB = pair[1]             
        partB = BODY_PARTS[partB]   
        
        if points[partA] and points[partB]:
            cv2.line(img, points[partA], points[partB], (0, 255, 0), 2)
    
    return img, points

# =============== cosine similarity ================
# 코사인 유사도 계산 함수
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1.T, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    if (norm_vector1 * norm_vector2) == 0:
        similarity = 1
    else: similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def get_pose_pair_vectors(gt_vectors, pose_pairs):
    pair_vectors = {}
    for pair in pose_pairs:
        part1_idx, part2_idx = pair
        if gt_vectors.get(part1_idx) is None or gt_vectors.get(part2_idx) is None:
            pair_vectors[tuple(pair)] = None
        else:
            pair_vectors[tuple(pair)] = np.array(gt_vectors[part1_idx]) - np.array(gt_vectors[part2_idx])
    return pair_vectors

def calculate_similarity_between_pose_pairs(pair_vectors_gt, pair_vectors_usr):
    similarities = {}
    for pair in pair_vectors_gt.keys():
        vector_gt = pair_vectors_gt[pair]
        vector_usr = pair_vectors_usr[pair]
        if vector_gt is None or vector_usr is None:
            similarities[pair] = None
        else:
            similarity = cosine_similarity(vector_gt, vector_usr)
            similarities[pair] = similarity
    return similarities

def calculate_average_similarity(similarities):
    valid_similarities = [similarity for similarity in similarities.values() if similarity is not None]
    if len(valid_similarities) == 0:
        return None
    return np.mean(valid_similarities)

def normalize_vector(vector):
    # 벡터의 크기 계산
    magnitude = np.linalg.norm(vector)

    # 벡터를 L2 정규화하여 크기를 맞춤
    normalized_vector = vector / magnitude if magnitude != 0 else vector

    return normalized_vector

def get_valid_vectors(gt_points, usr_points):
    valid_gt_vectors = {}
    valid_usr_vectors = {}

    # 두 리스트에서 None이 아닌 인덱스의 좌표를 가져와서 벡터를 만듦
    for idx, (gt, usr) in enumerate(zip(gt_points, usr_points)):
        if gt is not None and usr is not None:
            # 두 점을 이용하여 벡터 생성
            gt_vector = np.array(gt)
            usr_vector = np.array(usr)

            # L2 정규화
            gt_normalized = normalize_vector(gt_vector)
            usr_normalized = normalize_vector(usr_vector)

            # 정규화된 벡터 리스트에 추가
            valid_gt_vectors[idx] = gt_normalized
            valid_usr_vectors[idx] = usr_normalized

    return valid_gt_vectors, valid_usr_vectors

# =============== OKS =================
def keypoint_similarity(gt_kp, usr_kp, sigmas, areas):
    EPSILON = torch.finfo(torch.float32).eps
    k = 2*sigmas
    denom = 2 * (k**2) * (areas + EPSILON) #이때 areas[:,None,None] 19, 크기
    gt_kp.pop(1)
    usr_kp.pop(1)
    count = sum(1 for kp in usr_kp if kp is not None and isinstance(kp, tuple))-1
    for i in range(len(gt_kp)):
        if gt_kp[i] is None or usr_kp[i] is None:
            gt_kp[i] = (0, 0)
            usr_kp[i] = (0, 0)
    gt_kp = gt_kp[:-1]
    usr_kp = usr_kp[:-1]
    gt_kp = np.array(gt_kp)
    usr_kp = np.array(usr_kp)
    diff = gt_kp - usr_kp # 17x2
    squared_diff = np.square(diff) #17x2
    distance = np.sum(squared_diff, axis=1) # 17,
    exp_term = distance / denom
    weights = torch.where(exp_term == 0, torch.tensor(0.), torch.tensor(1.))
    oks = (torch.exp(-exp_term) * weights).sum() / (count + EPSILON)
    return oks

def resize_keypoints(image, keypoints, target_size=(368,368)):
    # Resize image
    resized_image = cv2.resize(image, target_size)

    # Calculate resize ratio
    resize_ratio = (target_size[0] / image.shape[1], target_size[1] / image.shape[0])

    # Resize keypoints
    resized_keypoints = []
    for kp in keypoints:
        if kp is None:
            resized_keypoints.append(None)
        else:
            # Resize each keypoint
            x = int(kp[0] * resize_ratio[0])
            y = int(kp[1] * resize_ratio[1])
            resized_keypoints.append((x, y))

    return resized_keypoints

# =============== using GT img =================
gt_img = cv2.imread(gt_img_path)
gt_img, gt_kp = output_keypoints_with_lines_image(gt_img)
cv2.imshow("gt", gt_img)

# =============== using webcam =================
capture = cv2.VideoCapture(0) #카메라 정보 받아옴

while cv2.waitKey(1) <0:  #아무 키나 누르면 끝난다.
    hasFrame, frame = capture.read()
    
    if not hasFrame:
        cv2.waitKey()
        break

    frame, usr_kp = output_keypoints_with_lines_image(frame)

    # 코사인 유사도 계산
    gt_vectors, usr_vectors = get_valid_vectors(gt_kp, usr_kp)
    pose_pair_vectors_gt = get_pose_pair_vectors(gt_vectors, POSE_PAIRS_COCO)
    pose_pair_vectors_usr = get_pose_pair_vectors(usr_vectors, POSE_PAIRS_COCO)
    similarities = calculate_similarity_between_pose_pairs(pose_pair_vectors_gt, pose_pair_vectors_usr)
    cos_avg_similarity = calculate_average_similarity(similarities)

    # oks 계산
    KPTS_OKS_SIGMAS_COCO = torch.tensor([.26, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, 35, .35])/10.0
    resized_gt_kp = resize_keypoints(gt_img, gt_kp)
    resized_usr_kp = resize_keypoints(frame, usr_kp)
    resized_gt_kp2=resized_gt_kp.copy()
    resized_usr_kp2=resized_usr_kp.copy()

    ## oks 계산에 필요한 area 계산 ##
    gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
    oks_coco = keypoint_similarity(resized_gt_kp2,
                                resized_usr_kp2,
                                sigmas=KPTS_OKS_SIGMAS_COCO,
                                areas=area)

    # 실시간으로 유사도 표시
    # extracted keypoints minimum
    count_none = sum(1 for kp in resized_usr_kp if kp is None)
    if count_none >= 6:
        print("검출된 키포인트 수가 부족합니다. 다시 준비해오세요!")
    else:
        cv2.putText(frame, f"Cosine Similarity: {cos_avg_similarity:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
            lineType=cv2.LINE_AA)
        cv2.putText(frame, f"OKS: {oks_coco:.2f}", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)
        cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()  #카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows() #모든 윈도우 창 닫기