# Apache Spark Optimization Guide

## Overview
Apache Spark is a unified analytics engine for large-scale data processing. This guide covers optimization techniques to maximize performance and efficiency.

## Spark Architecture

### Core Components
1. **Driver**: Coordinates the execution of the Spark application
2. **Executors**: Worker nodes that execute tasks and store data
3. **Cluster Manager**: Allocates resources across applications
4. **SparkContext**: Entry point for Spark functionality

### Execution Model
- **Jobs**: High-level operations triggered by actions
- **Stages**: Sets of tasks that can run in parallel
- **Tasks**: Individual units of work sent to executors
- **Shuffles**: Data exchange between stages

## Memory Management

### Memory Regions
1. **Execution Memory**: For computations in shuffles, joins, sorts, and aggregations
2. **Storage Memory**: For caching and propagating internal data
3. **User Memory**: For user data structures and internal metadata
4. **Reserved Memory**: For system use

### Configuration Parameters
```python
# Executor memory settings
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryFraction", "0.8")
spark.conf.set("spark.executor.memoryStorageFraction", "0.5")

# Driver memory settings
spark.conf.set("spark.driver.memory", "4g")
spark.conf.set("spark.driver.maxResultSize", "2g")
```

## Performance Optimization Techniques

### 1. Data Serialization
Choose efficient serialization formats to reduce I/O overhead.

```python
# Use Kryo serializer for better performance
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
spark.conf.set("spark.kryo.registrationRequired", "true")

# Register custom classes
spark.conf.set("spark.kryo.classesToRegister", "com.example.MyClass")
```

### 2. Partitioning Strategies
Proper partitioning is crucial for performance and parallelism.

```python
# Optimal partition size: 128MB - 1GB per partition
# Rule of thumb: 2-4 partitions per CPU core

# Repartition for even distribution
df_repartitioned = df.repartition(200, "user_id")

# Coalesce to reduce partitions (no shuffle)
df_coalesced = df.coalesce(50)

# Custom partitioner for specific use cases
from pyspark.sql.functions import hash, mod
df_custom = df.withColumn("partition_id", mod(hash("user_id"), 100))
```

### 3. Caching and Persistence
Cache frequently accessed DataFrames to avoid recomputation.

```python
from pyspark import StorageLevel

# Cache in memory (default)
df.cache()

# Persist with specific storage level
df.persist(StorageLevel.MEMORY_AND_DISK_SER)

# Unpersist when no longer needed
df.unpersist()

# Check cached tables
spark.catalog.isCached("table_name")
```

### 4. Predicate Pushdown
Filter data as early as possible to reduce processing volume.

```python
# Good: Filter early
df_filtered = spark.read.parquet("data.parquet") \
    .filter(col("date") >= "2023-01-01") \
    .filter(col("status") == "active")

# Bad: Filter after expensive operations
df_bad = spark.read.parquet("data.parquet") \
    .groupBy("user_id").sum("amount") \
    .filter(col("sum(amount)") > 1000)
```

### 5. Column Pruning
Select only necessary columns to reduce I/O and memory usage.

```python
# Good: Select specific columns
df_selected = df.select("user_id", "amount", "timestamp")

# Bad: Select all columns when only few are needed
df_all = df.select("*")
```

## Advanced Optimization Techniques

### 1. Adaptive Query Execution (AQE)
Enable AQE for automatic optimization based on runtime statistics.

```python
# Enable AQE
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
```

### 2. Broadcast Variables
Use broadcast variables for small datasets used across multiple tasks.

```python
# Broadcast small lookup table
lookup_dict = {"key1": "value1", "key2": "value2"}
broadcast_lookup = spark.sparkContext.broadcast(lookup_dict)

def map_function(row):
    return broadcast_lookup.value.get(row.key, "unknown")

# Use in transformations
df_mapped = df.rdd.map(map_function).toDF()
```

### 3. Bucketing
Pre-partition data to avoid shuffles in joins and aggregations.

```python
# Create bucketed table
df.write \
    .bucketBy(10, "user_id") \
    .sortBy("timestamp") \
    .saveAsTable("bucketed_table")

# Join bucketed tables (no shuffle)
bucketed_df1 = spark.table("bucketed_table1")
bucketed_df2 = spark.table("bucketed_table2")
joined = bucketed_df1.join(bucketed_df2, "user_id")
```

### 4. Dynamic Partition Pruning
Automatically eliminate partitions during query execution.

```python
# Enable dynamic partition pruning
spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly", "false")
```

## Join Optimization

### Join Types and Strategies
1. **Broadcast Hash Join**: Small table broadcasted to all nodes
2. **Sort Merge Join**: Both tables sorted and merged
3. **Shuffle Hash Join**: Data shuffled and hash-joined

```python
from pyspark.sql.functions import broadcast

# Force broadcast join for small table
df_large.join(broadcast(df_small), "key")

# Provide join hints
df1.hint("broadcast").join(df2, "key")
df1.join(df2.hint("shuffle_hash"), "key")
```

### Join Optimization Tips
- Broadcast smaller tables (< 10MB by default)
- Ensure join keys are not null
- Use appropriate join types (inner, left, right, full)
- Consider bucketing for frequently joined tables

## File Format Optimization

### Parquet Best Practices
```python
# Optimize Parquet writing
df.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .option("spark.sql.parquet.compression.codec", "snappy") \
    .parquet("output_path")

# Enable vectorized reading
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
spark.conf.set("spark.sql.parquet.columnarReaderBatchSize", "4096")
```

### Delta Lake Optimization
```python
# Optimize Delta tables
from delta.tables import DeltaTable

# Optimize file sizes
deltaTable = DeltaTable.forPath(spark, "delta_table_path")
deltaTable.optimize().executeCompaction()

# Z-ordering for better query performance
deltaTable.optimize().executeZOrderBy("user_id", "timestamp")

# Vacuum old files
deltaTable.vacuum(168)  # Retain 7 days of history
```

## Monitoring and Debugging

### Spark UI Analysis
- **Jobs**: Monitor job duration and failures
- **Stages**: Identify slow stages and data skew
- **Storage**: Check cached data and memory usage
- **Executors**: Monitor executor health and resource usage
- **SQL**: Analyze query execution plans

### Key Metrics to Monitor
1. **Task Duration**: Identify slow tasks and data skew
2. **Shuffle Read/Write**: Monitor data movement between stages
3. **GC Time**: Watch for memory pressure and GC overhead
4. **Executor Utilization**: Ensure efficient resource usage

### Common Performance Issues

#### Data Skew
```python
# Detect skew
df.groupBy("partition_key").count().orderBy(desc("count")).show()

# Mitigate skew with salting
from pyspark.sql.functions import rand, floor

df_salted = df.withColumn("salt", floor(rand() * 10))
df_salted = df_salted.withColumn("salted_key", 
                                concat(col("partition_key"), lit("_"), col("salt")))
```

#### Small Files Problem
```python
# Coalesce to reduce number of files
df.coalesce(10).write.parquet("output_path")

# Use appropriate partition size
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")  # 128MB
```

## Resource Allocation

### Cluster Sizing
```python
# Example configuration for medium workload
spark_config = {
    "spark.executor.instances": "20",
    "spark.executor.cores": "4",
    "spark.executor.memory": "8g",
    "spark.executor.memoryOverhead": "1g",
    "spark.driver.memory": "4g",
    "spark.driver.cores": "2"
}
```

### Dynamic Allocation
```python
# Enable dynamic allocation
spark.conf.set("spark.dynamicAllocation.enabled", "true")
spark.conf.set("spark.dynamicAllocation.minExecutors", "1")
spark.conf.set("spark.dynamicAllocation.maxExecutors", "100")
spark.conf.set("spark.dynamicAllocation.initialExecutors", "10")
```

This comprehensive guide provides the foundation for optimizing Spark applications across different scenarios and workloads.
