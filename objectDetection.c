#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "darknet.h"

#define SHM_NAME "/ipc_yolov4_shm"
#define MAX_BOXES 10

typedef struct
{
    int class_id;
    float confidence;
    int x, y, w, h;
} Detection;

typedef struct
{
    int count;
    Detection det[MAX_BOXES];
} SharedData;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    char *image_path = argv[1];

    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0)
    {
        perror("shm_open");
        return 1;
    }

    ftruncate(shm_fd, sizeof(SharedData));

    SharedData *shared = (SharedData *)mmap(
        NULL,
        sizeof(SharedData),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        shm_fd,
        0);

    if (shared == MAP_FAILED)
    {
        perror("mmap");
        return 1;
    }

    memset(shared, 0, sizeof(SharedData));

    network *net = load_network(
        "yolov4-tiny.cfg",
        "yolov4-tiny.weights",
        0);

    set_batch_network(net, 1);

    image im = load_image_color(image_path, 0, 0);

    network_predict_image(net, im);

    int nboxes = 0;
    float thresh = 0.5;

    detection *dets = get_network_boxes(
        net,
        im.w,
        im.h,
        thresh,
        0,
        0,
        1,
        &nboxes,
        0);

    do_nms_sort(
        dets,
        nboxes,
        net->layers[net->n - 1].classes,
        0.45);

    shared->count = 0;

    for (int i = 0; i < nboxes && shared->count < MAX_BOXES; i++)
    {
        for (int j = 0; j < net->layers[net->n - 1].classes; j++)
        {

            if (j == 0 && dets[i].prob[j] > thresh)
            {

                box b = dets[i].bbox;

                int left = (b.x - b.w / 2.0) * im.w;
                int top = (b.y - b.h / 2.0) * im.h;
                int width = b.w * im.w;
                int height = b.h * im.h;

                Detection *d = &shared->det[shared->count];

                d->class_id = j;
                d->confidence = dets[i].prob[j];
                d->x = left;
                d->y = top;
                d->w = width;
                d->h = height;

                shared->count++;
            }
        }
    }

    free_detections(dets, nboxes);
    free_image(im);

    printf("Detections written to shared memory. Count = %d\n", shared->count);

    return 0;
}
