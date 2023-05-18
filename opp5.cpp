#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <vector>

#define L 1000
#define LISTS_COUNT 10
#define TASK_COUNT 170000
#define MIN_TASKS_TO_SHARE 2
#define EXECUTOR_FINISHED_WORK -1
#define SENDING_TASKS 656
#define SENDING_TASK_COUNT 787
#define NO_TASKS_TO_SHARE -565

struct GlobalElements{
    pthread_t threads[2];
    pthread_mutex_t mutex;
    int * tasks;
    double SummaryDisbalance = 0;
    bool FinishedExecution = false;

    int ProcessCount;
    int ProcessRank;
    int RemainingTasks;
    int ExecutedTasks;
    int AdditionalTasks;
    double globalRes = 0;
};

void initializeTaskSet(int * taskSet, int taskCount, int iterCounter, void * glp) {
    GlobalElements * gl = (GlobalElements *)glp;
    for (int i = 0; i < taskCount; i++) {
        taskSet[i] = abs(50 - i%100)*abs(gl->ProcessRank - (iterCounter % gl->ProcessCount))*L;
    }
}

void executeTaskSet(int * taskSet, void * glp) {
    GlobalElements * gl = (GlobalElements *)glp;
    int i = 0;
    while (true){
        pthread_mutex_lock(&gl->mutex);
        if (i == gl->RemainingTasks) {
            pthread_mutex_unlock(&gl->mutex);
            break;
        }
        int weight = taskSet[i];
        i++;
        pthread_mutex_unlock(&gl->mutex);
        for (int j = 0; j < weight; j++) {
            gl->globalRes += cos(0.001488);
        }

        gl->ExecutedTasks++;
    }
    gl->RemainingTasks = 0;
}

void * AddTask(void * glp) {
    GlobalElements * gl = (GlobalElements *)glp;
    int ThreadResponse;
    for (int procIdx = 0; procIdx < gl->ProcessCount; procIdx++) {
        if (procIdx == gl->ProcessRank)
            continue;
        MPI_Send(&gl->ProcessRank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);
        MPI_Recv(&ThreadResponse, 1, MPI_INT, procIdx, SENDING_TASK_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (ThreadResponse == NO_TASKS_TO_SHARE)
             continue;
        gl->AdditionalTasks = ThreadResponse;
        memset(gl->tasks, 0, TASK_COUNT);
        MPI_Recv(gl->tasks, gl->AdditionalTasks, MPI_INT, procIdx, SENDING_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pthread_mutex_lock(&gl->mutex);
        gl->RemainingTasks = gl->AdditionalTasks;
        pthread_mutex_unlock(&gl->mutex);
        executeTaskSet(gl->tasks, gl);
    }
    pthread_exit(nullptr);
}

void* ExecutorStartRoutine(void * glp) {
    GlobalElements * gl = (GlobalElements *)glp;
    gl->tasks = new int[TASK_COUNT];
    double StartTime, FinishTime, IterationDuration, ShortestIteration, LongestIteration;

    for (int i = 0; i < LISTS_COUNT; i++) {
        StartTime = MPI_Wtime();
        initializeTaskSet(gl->tasks, TASK_COUNT, i, gl);
        gl->ExecutedTasks = 0;
        gl->RemainingTasks = TASK_COUNT;
        gl->AdditionalTasks = 0;
        executeTaskSet(gl->tasks, gl);
        AddTask(glp);
        FinishTime = MPI_Wtime();
        IterationDuration = FinishTime - StartTime;
        MPI_Allreduce(&IterationDuration, &LongestIteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&IterationDuration, &ShortestIteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        gl->SummaryDisbalance += (LongestIteration - ShortestIteration)/LongestIteration;
    }
    pthread_mutex_lock(&gl->mutex);
    gl->FinishedExecution = true;
    pthread_mutex_unlock(&gl->mutex);
    int Signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&Signal, 1, MPI_INT, gl->ProcessRank, 888, MPI_COMM_WORLD);
    pthread_exit(nullptr);
}

void* ReceiverStartRoutine(void * glp) {
    GlobalElements * gl = (GlobalElements *)glp;
    int AskingProcRank, Answer, PendingMessage;
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    while (!gl->FinishedExecution) {
        MPI_Recv(&PendingMessage, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD, &status);
        AskingProcRank = PendingMessage;
        pthread_mutex_lock(&gl->mutex);
        if (gl->RemainingTasks >= MIN_TASKS_TO_SHARE) {
            Answer = gl->RemainingTasks / (gl->ProcessCount*2);
            gl->RemainingTasks = gl->RemainingTasks / (gl->ProcessCount*2);
            MPI_Send(&Answer, 1, MPI_INT, AskingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
            MPI_Send(&gl->tasks[TASK_COUNT - Answer], Answer, MPI_INT, AskingProcRank, SENDING_TASKS, MPI_COMM_WORLD);

        } else {
            Answer = NO_TASKS_TO_SHARE;
            MPI_Send(&Answer, 1, MPI_INT, AskingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&gl->mutex);
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
    GlobalElements gl;
    MPI_Comm_rank(MPI_COMM_WORLD, &gl.ProcessRank);
    MPI_Comm_size(MPI_COMM_WORLD, &gl.ProcessCount);

    pthread_mutex_init(&gl.mutex, nullptr);
    pthread_attr_t ThreadAttributes;

    double start = MPI_Wtime();
    pthread_attr_init(&ThreadAttributes);
    pthread_attr_setdetachstate(&ThreadAttributes, PTHREAD_CREATE_JOINABLE);
    pthread_create(&gl.threads[0], &ThreadAttributes, ReceiverStartRoutine, &gl);
    pthread_create(&gl.threads[1], &ThreadAttributes, ExecutorStartRoutine, &gl);
    pthread_join(gl.threads[0], nullptr);
    pthread_join(gl.threads[1], nullptr);
    pthread_attr_destroy(&ThreadAttributes);
    pthread_mutex_destroy(&gl.mutex);

    if (gl.ProcessRank == 0) {
        std::cout << "Summary disbalance:" << gl.SummaryDisbalance/(LISTS_COUNT)*100 << "%" << std::endl;
        std::cout << "time taken: " << MPI_Wtime() - start << std::endl;
    }

    MPI_Finalize();
    return 0;
}
