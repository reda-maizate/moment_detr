import torch

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
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
    print(f'AWS region: {AWS_REGION}')
    # SQS_QUEUE_NAME = os.environ.get('SQS_QUEUE_NAME')
    print(f'SQS queue name: {SQS_QUEUE_NAME}')

    # Redis Environment Variables
    # REDIS_HOST = os.environ.get('REDIS_HOST')
    print(f'Redis host: {REDIS_HOST}')
    # REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
    print(f'Redis port: {REDIS_PORT}')
    # REDIS_USERNAME = os.environ.get('REDIS_USERNAME', "")
    print(f'Redis username: {REDIS_USERNAME}')
    # REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', "")
    print(f'Redis password: {REDIS_PASSWORD}')

    # Get the service resource
    sqs = boto3.resource('sqs', region_name=AWS_REGION, aws_access_key_id=ACCESS_ID, aws_secret_access_key=ACCESS_KEY)

    # Get the queue
    queue = sqs.get_queue_by_name(QueueName=SQS_QUEUE_NAME)#, QueueOwnerAWSAccountId=AWS_QUEUE_OWNER_ID)

    # Process messages by printing out body and optional author name
    timeout = time.time() + 60 * 3

    while True:
        if time.time() > timeout:
            break
        try:
            messages = queue.receive_messages(MessageAttributeNames=['All'], MaxNumberOfMessages=1, WaitTimeSeconds=5)
            counter = 0
            for message in messages:
                print('Consuming a message...')
                logger.info('Consuming a message...')
                # print("--------------------")
                # logger.info("--------------------")
                payload = json.loads(message.body)
                print(f'Payload: {payload}')
                logger.info(f'Payload: {payload}')
                bucket_name = payload.get('Records')[0].get('s3').get('bucket').get('name')
                print(f'Bucket name: {bucket_name}')
                logger.info(f'Bucket name: {bucket_name}')
                object_key = payload.get('Records')[0].get('s3').get('object').get('key')
                print(f'Object key: {object_key}')
                logger.info(f'Object key: {object_key}')
                message.delete()

                # TODO #1: Download videos from S3 bucket to local storage
                s3_client = boto3.client('s3', region_name=AWS_REGION, aws_access_key_id=ACCESS_ID,
                                         aws_secret_access_key=ACCESS_KEY)
                file_name = object_key.split('/')[-1]

                try:
                    s3_client.download_file(bucket_name, object_key, file_name)
                except Exception as e:
                    print("Error during download")
                    logger.error("Error during download")
                    print(e)
                    logger.error(e)

                print(f'Downloaded {object_key} from {bucket_name} to {file_name}')
                logger.info(f'Downloaded {object_key} from {bucket_name} to {file_name}')
                # print(f"Local files: {os.listdir()}")
                # logger.info(f'Local files: {os.listdir()}')

                # TODO #2: Run inference

                # TODO #5: Delete videos from S3 bucket and local storage
                try:
                    os.remove(file_name)
                except Exception as e:
                    print("Error during local deletion")
                    logger.error("Error during local deletion")
                    print(e)
                    logger.error(e)

                try:
                    s3_client.delete_object(Bucket=bucket_name, Key=object_key)
                except Exception as e:
                    print("Error during S3 deletion")
                    logger.error("Error during S3 deletion")
                    print(e)
                    logger.error(e)
                print(f'Deleted {object_key} from {bucket_name} and {file_name}')
                logger.info(f'Deleted {object_key} from {bucket_name} and {file_name}')

                # TODO #3: Push data to AWS ElastiCache (Redis) instance
                # logger.info("REDIS_HOST: " + REDIS_HOST)
                # print("REDIS_HOST: ", REDIS_HOST)
                # logger.info("REDIS_PORT: " + REDIS_PORT)
                # print("REDIS_PORT: ", REDIS_PORT)
                # logger.info("REDIS_USERNAME: " + REDIS_USERNAME)
                # print("REDIS_USERNAME: ", REDIS_USERNAME)
                # logger.info("REDIS_PASSWORD: " + REDIS_PASSWORD)
                # print("REDIS_PASSWORD: ", REDIS_PASSWORD)

                redis_cluster = RedisCluster(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME,
                                             password=REDIS_PASSWORD, decode_responses=True,
                                             skip_full_coverage_check=True, ssl=True)
                # TODO: Test avec RedisCluster bg
                if redis_cluster.ping():
                    print('Connected to Redis')
                    logger.info('Connected to Redis')
                else:
                    print('Could not connect to Redis')
                    logger.info('Could not connect to Redis')

                # TODO #4: Push data to AWS ElastiCache (Redis) cluster
                # Push list of elements in the key 'foo'
                key = f'project_id:{counter}'
                redis_cluster.delete(key)

                d = {"query": "test", "result": "test"}
                for k, v in d.items():
                    redis_cluster.hset(key, k, v)
                # redis_cluster.hset('foo', mapping={"query": "test", "result": "test"})
                print("Pushed data to Redis")
                logger.info("Pushed data to Redis")
                # Get the list of elements in the key 'foo'

                values_from_my_key = redis_cluster.hgetall('foo')
                for k, v in values_from_my_key.items():
                    print(k, v)
                    logger.info(k, v)

                redis_cluster.delete(key)  # TODO: Temporary delete for testing purposes
                counter += 1

        except Exception as e:
            logger.error(f"Error in message consuming: {e}")
            print(f"Error in message consuming: {e}")
            continue

    print('Done getting messages from SQS queue')
    logger.info('Done getting messages from SQS queue')


def run_example():
    # load example data
    from utils.basic_utils import load_jsonl
    video_path = "run_on_video/example/RoripwjYFp8_60.0_210.0.mp4"
    query_path = "run_on_video/example/queries.jsonl"
    queries = load_jsonl(query_path)
    query_text_list = [e["query"] for e in queries]
    ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"

    # run predictions
    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    # clip_model_name_or_path = "tmp/ViT-B-32.pt"
    moment_detr_predictor = MomentDETRPredictor(ckpt_path=ckpt_path, clip_model_name_or_path=clip_model_name_or_path,
                                                device="cpu")
    print("Run prediction...")
    predictions = moment_detr_predictor.localize_moment(video_path=video_path, query_list=query_text_list)

    # print data
    for idx, query_data in enumerate(queries):
        print("-" * 30 + f"idx{idx}")
        print(f">> query: {query_data['query']}")
        print(f">> video_path: {video_path}")
        # print(f">> GT moments: {query_data['relevant_windows']}")
        print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
              f"{predictions[idx]['pred_relevant_windows']}")
        # print(f">> GT saliency scores (only localized 2-sec clips): {query_data['saliency_scores']}")
        print(f">> Predicted saliency scores (for all 2-sec clip): "
              f"{predictions[idx]['pred_saliency_scores']}")


if __name__ == "__main__":
    main()
