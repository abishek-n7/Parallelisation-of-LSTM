#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef WINDOWS

#else
#include <unistd.h>
#endif

#include <omp.h>
#include "lstm.h"
#include "set.h"
#include "layers.h"
#include "utilities.h"

#include "std_conf.h"

#define ITERATIONS  100
#define NO_EPOCHS   0 

lstm_model_t *model = NULL, *layer1 = NULL, *layer2 = NULL;
lstm_model_t **model_layers;
lstm_model_parameters_t params;
set_t set;

static int write_output_directly_bytes = 0;
static char *read_network = NULL;
static char *seed = NULL;
static int store_after_training = 0;
static char save_model_folder_raw[256];
static char save_model_folder_json[256];

void store_the_net_layers(int signo)
{
  if ( SIGINT == signo ) {
    if ( model_layers != NULL ) {
      lstm_store(params.store_network_name_raw, &set,
      model_layers, params.layers);
      lstm_store_net_layers_as_json(model_layers, params.store_network_name_json, JSON_KEY_NAME_SET, &set, params.layers);
      printf("\nStored the net as: '%s'\nYou can use that file in the .html interface.\n", 
      params.store_network_name_json );
      printf("The net in its raw format is stored as: '%s'.\nYou can use that with the -r flag \
to continue refining the weights.\n", params.store_network_name_raw); 
    } else {
      printf("\nFailed to store the net!\n");
      exit(-1);
    }
  }

  exit(0);
  return;
}

void usage(char *argv[]) {
  printf("Usage: %s datafile [flag value]*\r\n", argv[0]);
  printf("\r\n");
  printf("Flags can be used to change the training procedure.\r\n");
  printf("The flags require a value to be passed as the following argument.\r\n");
  printf("    E.g., this is how you train with a learning rate set to 0.03:\r\n");
  printf("        %s datafile -lr 0.03\r\n", argv[0]);
  printf("\r\n");
  printf("The following flags are available:\r\n");
  printf("    -r  : read a previously trained network, the name of which is currently configured to be '%s'.\r\n", STD_LOADABLE_NET_NAME);
  printf("    -lr : learning rate that is to be used during training, see the example above.\r\n");
  printf("    -it : the number of iterations used for training (not to be confused with epochs).\r\n");
  printf("    -ep : the number of epochs used for training (not to be confused with iterations).\r\n");
  printf("    -mb : mini batch size.\r\n");
  printf("    -dl : decrease the learning rate over time, according to lr(n+1) <- lr(n) / (1 + n/value).\r\n");
  printf("    -st : number of iterations between how the network is stored during training. If 0 only stored once after training.\r\n");
  printf("    -out: number of characters to output directly, note: a network and a datafile must be provided.\r\n");
  printf("    -L  : Number of layers, may not exceed %d\r\n", LSTM_MAX_LAYERS);
  printf("    -N  : Number of neurons in every layer\r\n");
  printf("    -vr : Verbosity level. Set to zero and only the loss function after and not during training will be printed.\n");
  printf("    -c  : Don't train, only generate output. Seed given by the value. If -r is used, datafile is not considered.\r\n");
  printf("    -s  : Save folder, where models are stored (binary and JSON).\r\n");
  printf("\r\n");
  printf("Check std_conf.h to see what default values are used, these are set during compilation.\r\n");
  printf("\r\n");
  printf("%s compiled %s %s\r\n", argv[0], __DATE__, __TIME__);
  exit(1);
}

void parse_input_args(int argc, char** argv)
{
  int a = 0;

  while ( a < argc ) {

    if ( argc <= (a+1) )
      break; // All flags have values attributed to them

    if ( !strcmp(argv[a], "-r") ) {
      read_network = argv[a + 1];
    } else if ( !strcmp(argv[a], "-lr") ) {
      params.learning_rate = atof(argv[a + 1]);
      if ( params.learning_rate == 0.0 ) {
        usage(argv);
      }
    } else if ( !strcmp(argv[a], "-mb") ) {
      params.mini_batch_size = atoi(argv[a + 1]);
      if ( params.mini_batch_size <= 0 ) {
        usage(argv);
      }
    } else if ( !strcmp(argv[a], "-it") ) {
      params.iterations = (unsigned long) atol(argv[a + 1]);
      if ( params.iterations == 0 ) {
        usage(argv);
      }
    } else if ( !strcmp(argv[a], "-ep") ) {
      params.epochs = (unsigned long) atol(argv[a + 1]);
    } else if ( !strcmp(argv[a], "-dl") ) {
      params.learning_rate_decrease = atof(argv[a + 1]);
      if ( params.learning_rate_decrease == 0 ) {
        usage(argv);
      }
      params.decrease_lr = 1;
    } else if ( !strcmp(argv[a], "-st") ) {
      params.store_network_every = atoi(argv[a + 1]);
      if ( params.store_network_every == 0 ) {
        store_after_training = 1;
      }
    } else if ( !strcmp(argv[a], "-s") ) {
      memset(save_model_folder_json, 0, sizeof(save_model_folder_json));
      memset(save_model_folder_raw, 0, sizeof(save_model_folder_raw));

      snprintf(save_model_folder_json, sizeof(save_model_folder_json),
        "%s/%s", argv[a+1], STD_JSON_NET_NAME);
      snprintf(save_model_folder_raw, sizeof(save_model_folder_raw),
        "%s/%s", argv[a+1], STD_LOADABLE_NET_NAME);

      params.store_network_name_raw = save_model_folder_raw;
      params.store_network_name_json = save_model_folder_json;
    } else if ( !strcmp(argv[a], "-out") ) {
      write_output_directly_bytes = atoi(argv[a+1]);
      if ( write_output_directly_bytes <= 0 ) {
        usage(argv);
      }
    } else if ( !strcmp(argv[a], "-L") ) {
      params.layers = (unsigned int) atoi(argv[a+1]);
      if ( params.layers > LSTM_MAX_LAYERS ) {
        usage(argv);
      }
    } else if ( !strcmp(argv[a], "-N") ) {
      params.neurons = (unsigned int) atoi(argv[a+1]);
      if ( params.layers > LSTM_MAX_LAYERS ) {
        usage(argv);
      }
    } else if ( !strcmp(argv[a], "-vr") ) {
      params.print_progress = !!atoi(argv[a+1]);
    } else if ( !strcmp(argv[a], "-c") ) {
      seed = argv[a+1];
    }

    a += 2;
  }
}

static char * prettyPrintBytes(size_t bytes)
{
  static char buffer[128];
  const char *categories[4] = 
    {"B", "KB", "MB", "GB"};
  unsigned int category = 0;
  size_t displayBytes = bytes;
  size_t displayBytesRest = 0;

  while ( category < (sizeof(categories)/sizeof(*categories)) && displayBytes > 1024 )
  {
    displayBytesRest = displayBytes % 1024;
    displayBytes /= 1024;
    ++category;
  }

  snprintf(buffer,sizeof(buffer), "%zu.%zu %s", 
    displayBytes, displayBytesRest, categories[category]);
  return buffer;
}

double run_lstm_with_threads(int num_threads, const char *filename);

int main(int argc, char *argv[])
{
  int c;
  unsigned int file_size = 0, sz = 0;
  int *X_train, *Y_train;
  FILE *fp;

  memset(&params, 0, sizeof(params));

  params.iterations = ITERATIONS;
  params.epochs = NO_EPOCHS;
  params.loss_moving_avg = LOSS_MOVING_AVG;
  params.learning_rate = STD_LEARNING_RATE;
  params.momentum = STD_MOMENTUM;
  params.lambda = STD_LAMBDA;
  params.softmax_temp = SOFTMAX_TEMP;
  params.mini_batch_size = MINI_BATCH_SIZE;
  params.gradient_clip_limit = GRADIENT_CLIP_LIMIT;
  params.learning_rate_decrease = STD_LEARNING_RATE_DECREASE;
  params.stateful = STATEFUL;
  params.beta1 = 0.9;
  params.beta2 = 0.99;
  params.gradient_fit = GRADIENTS_FIT;
  params.gradient_clip = GRADIENTS_CLIP;
  params.decrease_lr = DECREASE_LR;
  params.model_regularize = MODEL_REGULARIZE;
  params.layers = LAYERS;
  params.neurons = NEURONS;
  params.optimizer = OPTIMIZE_ADAM;
  // Interaction configuration with the training of the network
  params.print_progress = PRINT_PROGRESS;
  params.print_progress_iterations = PRINT_EVERY_X_ITERATIONS;
  params.print_progress_sample_output = PRINT_SAMPLE_OUTPUT;
  params.print_progress_to_file = PRINT_SAMPLE_OUTPUT_TO_FILE;
  params.print_progress_number_of_chars = NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING;
  params.print_sample_output_to_file_arg = PRINT_SAMPLE_OUTPUT_TO_FILE_ARG;
  params.print_sample_output_to_file_name = PRINT_SAMPLE_OUTPUT_TO_FILE_NAME;
  params.store_progress_every_x_iterations = STORE_PROGRESS_EVERY_X_ITERATIONS;
  params.store_progress_file_name = PROGRESS_FILE_NAME;
  params.store_network_name_raw = STD_LOADABLE_NET_NAME;
  params.store_network_name_json = STD_JSON_NET_NAME;
  params.store_char_indx_map_name = JSON_KEY_NAME_SET;

  srand(time(NULL));

  if (argc < 2) {
    usage(argv);
    return -1;
  }

  parse_input_args(argc, argv);
  initialize_set(&set);

  // Define thread counts to test
  int processors[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
  int num_tests = sizeof(processors) / sizeof(processors[0]);
  double times[num_tests];

  // Open CSV file for results
  FILE *csv_fp = fopen("threads_vs_time.csv", "w");
  if (!csv_fp) {
    printf("Error opening CSV file\n");
    return -1;
  }
  fprintf(csv_fp, "Threads,Time (s)\n");

  // Run benchmarks for each thread count - start from 0 (1 thread) to ensure we test with 1 thread first
  for (int i = 1; i < num_tests; i++) {
    printf("Running benchmark with %d threads...\n", processors[i]);
    double exec_time = run_lstm_with_threads(processors[i], argv[1]);
    
    // Check if run was successful
    if (exec_time < 0) {
      printf("Error in benchmark with %d threads. Skipping remaining tests.\n", processors[i]);
      break;
    }
    
    times[i] = exec_time;
    printf("Threads: %d, Time: %f seconds\n", processors[i], exec_time);
    fprintf(csv_fp, "%d,%f\n", processors[i], exec_time);
    
    // Flush CSV file after each test to ensure data is saved even if there's a crash
    fflush(csv_fp);
  }

  fclose(csv_fp);
  printf("Results saved in threads_vs_time.csv\n");

  return 0;
}

double run_lstm_with_threads(int num_threads, const char *filename) {
  double start, end;
  int c;
  unsigned int file_size = 0, sz = 0;
  int *X_train, *Y_train;
  FILE *fp;
  
  // Read the file size once before parallelization
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Could not open file: %s\n", filename);
    return -1.0;
  }

  while ((c = fgetc(fp)) != EOF) {
    set_insert_symbol(&set, (char)c);
    ++file_size;
  }

  fclose(fp);

  printf("File size: %u characters, initializing training data...\n", file_size);
  
  // Allocate memory for training data
  X_train = calloc(file_size + 1, sizeof(int));
  if (X_train == NULL) {
    printf("Memory allocation failed for X_train!\n");
    return -1.0;
  }

  X_train[file_size] = X_train[0]; // Set the last element
  Y_train = &X_train[1]; // Y_train points to the second element

  // Load the data once
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Could not open file for data reading: %s\n", filename);
    free(X_train);
    return -1.0;
  }

  sz = 0;
  while ((c = fgetc(fp)) != EOF) {
    X_train[sz++] = set_char_to_indx(&set, c);
  }
  fclose(fp);

  printf("Training data loaded, starting parallel execution with %d threads...\n", num_threads);
  
  // Start timing for this run
  start = omp_get_wtime();

  // Parallel region - fixed to avoid race conditions
  #pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    lstm_model_t **thread_model_layers = NULL;
    set_t thread_set;
    unsigned int p = 0;
    double thread_loss = 0.0;
    
    // Initialize a set for this thread
    initialize_set(&thread_set);
    
    // Copy character set from file instead of directly accessing set structure
    FILE *thread_fp = fopen(filename, "r");
    if (thread_fp != NULL) {
      while ((c = fgetc(thread_fp)) != EOF) {
        set_insert_symbol(&thread_set, (char)c);
      }
      fclose(thread_fp);
      
      #pragma omp critical(printf)
      {
        printf("Thread %d: Initialized character set\n", thread_id);
      }
    } else {
      #pragma omp critical(printf)
      {
        printf("Thread %d: Failed to open file for character set initialization\n", thread_id);
      }
      // Skip this thread if file couldn't be opened
      goto thread_cleanup;
    }
    
    // Create thread-local copies of the training data
    int *thread_X_train = NULL;
    int *thread_Y_train = NULL;
    
    #pragma omp critical(data_copy)
    {
      thread_X_train = calloc(file_size + 1, sizeof(int));
      if (thread_X_train != NULL) {
        memcpy(thread_X_train, X_train, (file_size + 1) * sizeof(int));
        thread_Y_train = &thread_X_train[1];
      }
    }
    
    if (thread_X_train == NULL) {
      #pragma omp critical(printf)
      {
        printf("Thread %d: Failed to allocate memory for local training data\n", thread_id);
      }
      goto thread_cleanup;
    }
    
    if (read_network != NULL) {
      // Load existing network - critical section to prevent file access conflicts
      #pragma omp critical(network_load)
      {
        lstm_load(read_network, &thread_set, &params, &thread_model_layers);
      }
      
      // Check for new features
      int FRead = set_get_features(&thread_set);
      int FReadNewAfterDataFile = set_get_features(&thread_set);
      
      if (FReadNewAfterDataFile > FRead) {
        #pragma omp critical(printf)
        {
          printf("Thread %d: New features detected. Adjusting network layers.\n", thread_id);
        }
        lstm_reinit_model(
          thread_model_layers,
          params.layers,
          FRead,
          FReadNewAfterDataFile
        );
      }
    } else {
      // Initialize new model
      thread_model_layers = calloc(params.layers, sizeof(lstm_model_t*));
      
      if (thread_model_layers == NULL) {
        #pragma omp critical(printf)
        {
          printf("Thread %d: Error allocating memory for layers!\n", thread_id);
        }
        // Clean up thread resources and exit
        if (thread_X_train) free(thread_X_train);
        goto thread_cleanup;
      }
      
      // Initialize each layer
      p = 0;
      while (p < params.layers) {
        // All layers have the same training parameters
        int X;
        int N = params.neurons;
        int Y;

        if (params.layers == 1) {
          X = set_get_features(&thread_set);
          Y = set_get_features(&thread_set);
        } else {
          if (p == 0) {
            X = set_get_features(&thread_set);
            Y = params.neurons;
          } else if (p == params.layers - 1) {
            X = params.neurons;
            Y = set_get_features(&thread_set);
          } else {
            X = params.neurons;
            Y = params.neurons;
          }
        }

        #pragma omp critical(printf)
        {
          printf("Thread %d: Initializing layer %u with dimensions X=%d, N=%d, Y=%d\n", 
                 thread_id, p, X, N, Y);
        }

        lstm_init_model(X, N, Y, &thread_model_layers[p], 0, &params);
        ++p;
      }
    }
    
    // Now each thread has its own model, proceed with training
    if (thread_model_layers != NULL) {
      // Handle direct output or training
      if (write_output_directly_bytes && read_network != NULL) {
        #pragma omp critical(output)
        {
          lstm_output_string_layers(thread_model_layers, &thread_set, 
                                   set_indx_to_char(&thread_set, 0), 
                                   write_output_directly_bytes, params.layers);
        }
      } else if (write_output_directly_bytes && read_network == NULL) {
        #pragma omp critical(printf)
        {
          printf("Thread %d: Invalid state! Output requested without a network.\n", thread_id);
        }
      } else if (seed != NULL) {
        // Handle generating from seed
        #pragma omp critical(output)
        {
          lstm_output_string_from_string(thread_model_layers, &thread_set, 
                                       seed, params.layers, 256);
        }
      } else {
        // Handle training with thread-local data
        #pragma omp critical(printf)
        {
          printf("Thread %d: Training LSTM network...\n", thread_id);
        }
        
        // Each thread trains its own copy of the model with its own data copy
        lstm_train(
          thread_model_layers,
          &params,
          &thread_set,
          file_size,
          thread_X_train,  // Use thread-local copy
          thread_Y_train,  // Use thread-local copy
          params.layers,
          &thread_loss
        );
        
        #pragma omp critical(printf)
        {
          printf("Thread %d: Training completed successfully\n", thread_id);
        }
        
        // Store the model if needed
        if (store_after_training) {
          char thread_raw_name[256];
          char thread_json_name[256];
          
          snprintf(thread_raw_name, sizeof(thread_raw_name), 
                 "%s.thread%d", params.store_network_name_raw, thread_id);
          snprintf(thread_json_name, sizeof(thread_json_name), 
                 "%s.thread%d", params.store_network_name_json, thread_id);
          
          #pragma omp critical(storage)
          {
            lstm_store(thread_raw_name, &thread_set, thread_model_layers, params.layers);
            
            lstm_store_net_layers_as_json(thread_model_layers, thread_json_name, 
                                      JSON_KEY_NAME_SET, &thread_set, params.layers);
            
            printf("Thread %d: Loss after training: %lf\n", thread_id, thread_loss);
          }
        }
      }
      
      // Free thread resources
      if (thread_X_train) {
        free(thread_X_train);
        thread_X_train = NULL;
      }
      
      for (int i = 0; i < params.layers; i++) {
        if (thread_model_layers[i] != NULL) {
          lstm_free_model(thread_model_layers[i]);
          thread_model_layers[i] = NULL;
        }
      }
      
      free(thread_model_layers);
      thread_model_layers = NULL;
    }
    
thread_cleanup:
    #pragma omp critical(printf)
    {
      printf("Thread %d: Cleaning up resources\n", thread_id);
    }
  } // End of parallel region
  
  end = omp_get_wtime();
  double total_time = end - start;
  
  printf("Parallel execution completed in %.2f seconds\n", total_time);
  
  // Free resources for this run
  free(X_train);
  
  return total_time;
}