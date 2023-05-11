#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <cstring>

#define L 1000
#define LISTS_COUNT 500
#define TASK_COUNT 2000
#define MIN_TASKS_TO_SHARE 2
#define EXECUTOR_FINISHED_WORK -1
#define SENDING_TASKS 656
#define SENDING_TASK_COUNT 787
#define NO_TASKS_TO_SHARE -565

pthread_t threads[2];
pthread_mutex_t mutex;
int *tasks;
std::ofstream *LogFiles;

double SummaryDisbalance = 0;
bool FinishedExecution = false;

int ProcessCount;
int ProcessRank;
int RemainingTasks;
int ExecutedTasks;
int AdditionalTasks;
double globalRes = 0;

void initializeTaskSet(int *taskSet, int taskCount, int iterCounter) {
    for (int i = 0; i < taskCount; i++) {
        taskSet[i] = abs(i%100)*abs(ProcessRank - (iterCounter % ProcessCount))*L;
    }
}

void executeTaskSet(int *taskSet) {
    for(int i = 0; i < RemainingTasks; i++) {
        pthread_mutex_lock(&mutex);
        int weight = taskSet[i];
        pthread_mutex_unlock(&mutex);

        for(int j = 0; j < weight; j++) {
            globalRes += cos(0.001488);
        }

        ExecutedTasks++;
    }
    RemainingTasks = 0;
}

void* ExecutorStartRoutine(void * args) {
    args = args;
    tasks = new int[TASK_COUNT];
    double StartTime, FinishTime, IterationDuration, ShortestIteration, LongestIteration;

    for (int i = 0; i < LISTS_COUNT; i++) {
        StartTime = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Iteration " << i << ". Initializing tasks. " << std::endl;
        initializeTaskSet(tasks, TASK_COUNT, i);
        ExecutedTasks = 0;
        RemainingTasks = TASK_COUNT;
        AdditionalTasks = 0;

        executeTaskSet(tasks);
        std::cout << "Process " << ProcessRank << " executed tasks in " << MPI_Wtime() - StartTime << " Now requesting for some additional. " << std::endl;
        int ThreadResponse;

        for (int procIdx = 0; procIdx < ProcessCount; procIdx++) {
            if (procIdx != ProcessRank) {
                std::cout << "Process " << ProcessRank << " is asking " << procIdx << " for some tasks."<< std::endl;
                MPI_Send(&ProcessRank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);
                std::cout << "waiting for task count" << std::endl;
                MPI_Recv(&ThreadResponse, 1, MPI_INT, procIdx, SENDING_TASK_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "Process " << procIdx << " answered " << ThreadResponse << std::endl;

                if (ThreadResponse != NO_TASKS_TO_SHARE) {
                    AdditionalTasks = ThreadResponse;
                    memset(tasks, 0, TASK_COUNT);
                    std::cout << "waiting for tasks" << std::endl;
                    MPI_Recv(tasks, AdditionalTasks, MPI_INT, procIdx, SENDING_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pthread_mutex_lock(&mutex);
                    RemainingTasks = AdditionalTasks;
                    pthread_mutex_unlock(&mutex);
                    executeTaskSet(tasks);
                }
            }

        }
        FinishTime = MPI_Wtime();
        IterationDuration = FinishTime - StartTime;

        MPI_Allreduce(&IterationDuration, &LongestIteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&IterationDuration, &ShortestIteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "Process " << ProcessRank << " executed " << ExecutedTasks << " tasks. " << AdditionalTasks << " were additional." << std::endl;
        std::cout << "Cos sum is " << globalRes << ". Time taken: " << IterationDuration << std::endl;
        SummaryDisbalance += (LongestIteration - ShortestIteration)/LongestIteration;
        std::cout << "Max time difference: " << LongestIteration - ShortestIteration  << std::endl;
        std::cout << "Disbalance rate is " << ((LongestIteration - ShortestIteration)/ LongestIteration) * 100 << "%" << std::endl;
        LogFiles[ProcessRank] << IterationDuration << std::endl;
    }

    std::cout << "Proc " << ProcessRank << " finished iterations sending signal" << std::endl;
    pthread_mutex_lock(&mutex);
    FinishedExecution = true;
    pthread_mutex_unlock(&mutex);
    int Signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&Signal, 1, MPI_INT, ProcessRank, 888, MPI_COMM_WORLD);
    delete [] tasks;
    pthread_exit(nullptr);
}

void* ReceiverStartRoutine(void * args) {
    args = args;
    int AskingProcRank, Answer, PendingMessage;
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    while (!FinishedExecution) {
        MPI_Recv(&PendingMessage, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD, &status);

        if (PendingMessage == EXECUTOR_FINISHED_WORK) {
            std::cout << "Executor finished work on proc " << ProcessRank << std::endl;
        }
        AskingProcRank = PendingMessage;
        pthread_mutex_lock(&mutex);
        std::cout << "Process " << AskingProcRank << " requested tasks. I have " << RemainingTasks << " tasks now. " << std::endl;
        if (RemainingTasks >= MIN_TASKS_TO_SHARE) {
            Answer = RemainingTasks / (ProcessCount*2);
            RemainingTasks = RemainingTasks / (ProcessCount*2);

            std::cout << "Sharing " << Answer << " tasks. " << std::endl;

            MPI_Send(&Answer, 1, MPI_INT, AskingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
            MPI_Send(&tasks[TASK_COUNT - Answer], Answer, MPI_INT, AskingProcRank, SENDING_TASKS, MPI_COMM_WORLD);
        } else {
            Answer = NO_TASKS_TO_SHARE;
            MPI_Send(&Answer, 1, MPI_INT, AskingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&mutex);
    }

    pthread_exit(nullptr);
}


int main(int argc, char* argv[]) {
    int ThreadSupport;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ThreadSupport);
    if(ThreadSupport != MPI_THREAD_MULTIPLE) {
    std::cout << "Error" << std::endl;
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &ProcessRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcessCount);

    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t ThreadAttributes;

    LogFiles = new std::ofstream [ProcessCount];
    char* name = new char[12];
    for (int i = 0; i < ProcessCount; i++) {
        sprintf(name, "Log_%d.txt", i);
        LogFiles[i].open(name);
    }
    double start = MPI_Wtime();
    pthread_attr_init(&ThreadAttributes);
    pthread_attr_setdetachstate(&ThreadAttributes, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &ThreadAttributes, ReceiverStartRoutine, NULL);
    pthread_create(&threads[1], &ThreadAttributes, ExecutorStartRoutine, NULL);
    pthread_join(threads[0], nullptr);
    pthread_join(threads[1], nullptr);
    pthread_attr_destroy(&ThreadAttributes);
    pthread_mutex_destroy(&mutex);

    if (ProcessRank == 0) {
        std::cout << "Summary disbalance:" << SummaryDisbalance/(LISTS_COUNT)*100 << "%" << std::endl;
        std::cout << "time taken: " << MPI_Wtime() - start << std::endl;
    }

    MPI_Finalize();
    return 0;
}
