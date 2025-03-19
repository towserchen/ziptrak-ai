import time
from utils import gateway
from multiprocessing import Process, Queue, RLock, Value, Event

import boto3

lock = RLock()
is_idle = Value('b', True)
task_queue = Queue()
result_queue = Queue()
image_detection_event = Event()


def subprocess_detect_image(task_queue, result_queue, image_detection_event):
    import app as detect_app

    image_detection_event.set()

    while True:
        command = task_queue.get()
        print(f'[Detect] Command recv: {command}')

        additional_parameter = command['additional_parameter']
        result = detect_app.detect_file(command['file_path'], additional_parameter['is_indoor'], False, False)

        print(f'[Detect] Result: {result}')
        result_queue.put({
            'task_uuid': command['task_uuid'],
            'result': result
        })


def download_file_from_s3(s3_bucket, s3_key):
    s3 = boto3.client("s3")

    ext = s3_key.split('.')[-1]
    local_file_path = f'./uploads/{time.time()}.{ext}'
    s3.download_file(s3_bucket, s3_key, local_file_path)

    return local_file_path


# Add an image detection task to task queue
def add_image_task(task_queue, lock, is_idle, task_uuid, s3_bucket, file_key, additional_parameter=None):
    if not is_idle.value:
        return False

    with lock:
        is_idle.value = False
        print(f'[AddTask] is_idle: {is_idle.value}')

    print(f'[AddTask] s3_bucket: {s3_bucket}, file_key: {file_key}')
    file_path = download_file_from_s3(s3_bucket, file_key)
    
    #file_path = 'samples/1.jpg' # For test

    print(f'[AddTask] Dispatch task: {file_key} with additional parameter: {additional_parameter}')
    
    task_queue.put({
        'task_uuid': task_uuid,
        'file_path': file_path,
        'additional_parameter': additional_parameter
    })

    return True


if __name__ == '__main__':
    detect_process = Process(target=subprocess_detect_image, args=(task_queue, result_queue, image_detection_event))
    detect_process.start()

    image_detection_event.wait()

    while True:
        try:
            data = result_queue.get(block=False)
            
            with lock:
                is_idle.value = True

            print(f'[Result] Result recv: {data}')
            print(f'[Result] is_idle: {is_idle.value}')

            coordinate_list = []

            for coordinate in data['result']:
                _coordinate = [[int(x) for x in point] for point in coordinate]
                coordinate_list.append(_coordinate)

            gateway.report(data['task_uuid'], coordinate_list)
        except:
            pass

        if is_idle.value:
            # Get a task
            new_task = gateway.get_task()
            print(f'[Result] New task: {new_task}')

            if new_task:
                print(f'[Result] Add task: {new_task}')

                try:
                    add_image_task(task_queue, lock, is_idle, new_task['uuid'], new_task['s3_bucket'], new_task['file_key'], new_task['additional_parameter'])
                except Exception as e:
                    with lock:
                        is_idle.value = True

                    print(f'[Result] exception: {e}')
        else:
            pass # For test
            #print(f'[Result] Detection in progress, is_idle: {is_idle.value}')