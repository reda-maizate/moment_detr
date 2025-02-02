import torch

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

from utils.basic_utils import l2_normalize_np_array
import torch.nn.functional as F
import numpy as np
import boto3
import json
import os
import time
import logging
import redis
from rediscluster import RedisCluster

AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
REDIS_USERNAME = os.environ.get('REDIS_USERNAME')
SQS_QUEUE_NAME = os.environ.get('SQS_QUEUE_NAME')
ACCESS_ID = os.environ.get('ACCESS_ID')
ACCESS_KEY = os.environ.get('ACCESS_KEY')
AWS_QUEUE_OWNER_ID = os.environ.get('AWS_QUEUE_OWNER_ID')


class MomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(framerate=1 / self.clip_len, size=224, centercrop=True,
                                                      model_name_or_path=clip_model_name_or_path, device=device)
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding of this pretrained MomentDETR only support video up " \
                               "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(query_feats, dtype=torch.float32, device=self.device,
                                                   fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(src_vid=video_feats, src_vid_mask=video_mask, src_txt=query_feats, src_txt_mask=query_mask)

        # decode outputs
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in MomentDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(query=query_list[idx],  # str
                                  vid=video_path, pred_relevant_windows=cur_ranked_preds,
                                  # List([st(float), ed(float), score(float)])
                                  pred_saliency_scores=saliency_scores[idx]
                                  # List(float), len==n_frames, scores for each frame
                                  )
            predictions.append(cur_query_pred)

        return predictions


def main():
    print('Starting...')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info('Starting...')

    # SQS Environment Variables
    # AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    # print(f'AWS region: {AWS_REGION}')
    # SQS_QUEUE_NAME = os.environ.get('SQS_QUEUE_NAME')
    # print(f'SQS queue name: {SQS_QUEUE_NAME}')

    # Redis Environment Variables
    # REDIS_HOST = os.environ.get('REDIS_HOST')
    # print(f'Redis host: {REDIS_HOST}')
    # REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
    # print(f'Redis port: {REDIS_PORT}')
    # REDIS_USERNAME = os.environ.get('REDIS_USERNAME', "")
    # print(f'Redis username: {REDIS_USERNAME}')
    # REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', "")
    # print(f'Redis password: {REDIS_PASSWORD}')

    sqs = boto3.resource('sqs', region_name=AWS_REGION)

    # Get the queue
    queue = sqs.get_queue_by_name(QueueName=SQS_QUEUE_NAME)

    # Process messages by printing out body and optional author name
    # timeout = time.time() + 60 * 3

    while True:
        # if time.time() > timeout:
        #     break
        try:
            messages = queue.receive_messages(MessageAttributeNames=['All'], MaxNumberOfMessages=1, WaitTimeSeconds=5)
            for message in messages:
                print("Consuming message...")
                bucket_name, project_id, object_key = parse_message(message)

                if bucket_name is None and project_id is None and object_key is None:
                    continue

                #1: Download videos from S3 bucket to local storage
                s3_client = boto3.client('s3', region_name=AWS_REGION, aws_access_key_id=ACCESS_ID,
                                         aws_secret_access_key=ACCESS_KEY)
                file_name = download_video_from_bucket(s3_client, bucket_name, object_key)

                #2: Run inference
                print("Running inference...")
                res = run_inference(project_id, file_name)
                print("Inference done")

                #3: Delete videos from S3 bucket and local storage
                delete_video_in_bucket_and_locally(s3_client, bucket_name, object_key, file_name)

                #4: Push data to AWS MemoryDB (Redis) instance
                connect_and_push_to_redis(res, REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD)
                print("------------------------------------")

        except Exception as e:
            print(f"Error in message consuming: {e}")
            continue

    print('Done getting messages from SQS queue')


def run_inference(project_id, file_name):
    # load example data
    from utils.basic_utils import load_jsonl
    category = file_name.split('_')[1]
    print(f"Category: {category}")
    video_path = f"{file_name}"

    if category == 'lol':
        query_path = "run_on_video/example/queries_lol.jsonl"
        ckpt_path = "run_on_video/moment_detr_ckpt/model_best_lol.ckpt"
    else:
        query_path = "run_on_video/example/queries.jsonl"
        ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"

    queries = load_jsonl(query_path)
    query_text_list = [e["query"] for e in queries]


    # run predictions
    # print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    # clip_model_name_or_path = "tmp/ViT-B-32.pt"
    moment_detr_predictor = MomentDETRPredictor(ckpt_path=ckpt_path, clip_model_name_or_path=clip_model_name_or_path,
                                                device="cpu")
    print("Run prediction...")
    predictions = moment_detr_predictor.localize_moment(video_path=video_path, query_list=query_text_list)

    list_of_results = []

    for idx, query_data in enumerate(queries):
        res = {}
        res[f"{project_id}:{video_path.split('/')[-1]}:{query_data['query']}"] = {
            "pred_moments": predictions[idx]['pred_relevant_windows'],
            "pred_saliency_scores": predictions[idx]['pred_saliency_scores']}
        list_of_results.append(res)

    pprint(list_of_results)
    return list_of_results


def parse_message(message):
    payload = json.loads(message.body)

    if payload.get('Event', None) is not None:
        return None, None, None
    print('Detected an insertion in Splitted S3 bucket...')
    # print(f'Payload: {payload}')
    bucket_name = payload.get('Records')[0].get('s3').get('bucket').get('name')
    print(f'Bucket name: {bucket_name}')

    object_key = payload.get('Records')[0].get('s3').get('object').get('key')
    project_id, video_name = object_key.split('/')
    print(f'Project id: {project_id}, video name: {video_name}, object key: {object_key}')

    message.delete()
    return bucket_name, project_id, object_key


def download_video_from_bucket(s3_client, bucket_name, object_key):
    file_name = object_key.split('/')[-1]

    try:
        s3_client.download_file(bucket_name, object_key, file_name)
    except Exception as e:
        print("Error during download")
        print(e)

    print(f'Downloaded {object_key} from {bucket_name} to {file_name}')
    return file_name


def delete_video_in_bucket_and_locally(s3_client, bucket_name, object_key, file_name):
    try:
        os.remove(file_name)
    except Exception as e:
        print("Error during local deletion")
        print(e)

    # try:
    #     s3_client.delete_object(Bucket=bucket_name, Key=object_key)
    # except Exception as e:
    #     print("Error during S3 deletion")
    #     print(e)

    print(f'Deleted {object_key} from {bucket_name} and {file_name}')


def connect_and_push_to_redis(res, host, port, username, password):
    redis_cluster = RedisCluster(host=host, port=port, username=username,
                                 password=password, decode_responses=True,
                                 skip_full_coverage_check=True, ssl=True)

    if redis_cluster.ping():
        print('Connected to Redis')
    else:
        print('Could not connect to Redis')

    #4: Push data to AWS MemoryDB (Redis) cluster
    for elm in res:
        for k, v in elm.items():
            redis_cluster.delete(k)
            redis_cluster.set(k, str(v))
            redis_cluster.expire(k, 3600)
            print("Inserted data to Redis key: ", k, "with value: ", v)

    # Get the list of elements in the key 'foo'
    for elm in res:
        for k, v in elm.items():
            print(f"Get values from Redis key: {k}, with value: {redis_cluster.get(k)}")
            # redis_cluster.delete(k)  # TODO: Temporary delete for testing purposes


if __name__ == "__main__":
    main()
