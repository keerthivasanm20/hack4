# Data Engineering Best Practices

## Code Quality and Development

### 1. Version Control and Collaboration
- Use Git for version control of all data engineering code
- Implement branching strategies (GitFlow, GitHub Flow)
- Write meaningful commit messages with context
- Use pull requests for code review and collaboration
- Tag releases for production deployments

### 2. Code Organization and Structure
```python
# Project structure example
data_engineering_project/
├── src/
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── ingestion/
│   │   ├── transformation/
│   │   └── loading/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── helpers.py
│   └── tests/
├── config/
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── docs/
├── requirements.txt
├── setup.py
└── README.md
```

### 3. Configuration Management
- Externalize all configuration parameters
- Use environment-specific configuration files
- Never hardcode credentials or sensitive information
- Use configuration validation and type checking

```python
# Example configuration class
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    
@dataclass
class PipelineConfig:
    batch_size: int
    retry_attempts: int
    timeout_seconds: int
    database: DatabaseConfig
    
def load_config(config_path: str) -> PipelineConfig:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return PipelineConfig(**config_data)
```

## Data Quality and Validation

### 1. Data Quality Framework
Implement comprehensive data quality checks at multiple stages.

```python
from typing import List, Dict, Any
from enum import Enum

class DataQualityRule:
    def __init__(self, name: str, description: str, severity: str):
        self.name = name
        self.description = description
        self.severity = severity
    
    def validate(self, data) -> bool:
        raise NotImplementedError

class NotNullRule(DataQualityRule):
    def __init__(self, column: str):
        super().__init__(
            name=f"not_null_{column}",
            description=f"Column {column} should not contain null values",
            severity="error"
        )
        self.column = column
    
    def validate(self, df) -> bool:
        null_count = df.filter(df[self.column].isNull()).count()
        return null_count == 0

class RangeRule(DataQualityRule):
    def __init__(self, column: str, min_val: float, max_val: float):
        super().__init__(
            name=f"range_{column}",
            description=f"Column {column} should be between {min_val} and {max_val}",
            severity="warning"
        )
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, df) -> bool:
        out_of_range = df.filter(
            (df[self.column] < self.min_val) | 
            (df[self.column] > self.max_val)
        ).count()
        return out_of_range == 0
```

### 2. Schema Evolution and Validation
- Implement schema versioning for backward compatibility
- Use schema registries for centralized schema management
- Validate schema changes before deployment
- Handle schema evolution gracefully in pipelines

### 3. Data Lineage and Documentation
- Track data lineage from source to destination
- Document data transformations and business logic
- Maintain data dictionaries and catalog
- Implement automated documentation generation

## Testing Strategies

### 1. Unit Testing
Test individual functions and transformations in isolation.

```python
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

class TestDataTransformations:
    @pytest.fixture(scope="class")
    def spark_session(self):
        return SparkSession.builder \
            .appName("testing") \
            .master("local[2]") \
            .getOrCreate()
    
    def test_user_aggregation(self, spark_session):
        # Arrange
        schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("amount", IntegerType(), True)
        ])
        
        test_data = [("user1", 100), ("user1", 200), ("user2", 150)]
        df = spark_session.createDataFrame(test_data, schema)
        
        # Act
        result = aggregate_user_amounts(df)
        
        # Assert
        expected_data = [("user1", 300), ("user2", 150)]
        expected_df = spark_session.createDataFrame(expected_data, 
                                                   ["user_id", "total_amount"])
        
        assert result.collect() == expected_df.collect()
```

### 2. Integration Testing
Test complete pipeline workflows end-to-end.

```python
def test_pipeline_integration(spark_session, temp_dir):
    # Setup test data
    input_path = f"{temp_dir}/input"
    output_path = f"{temp_dir}/output"
    
    create_test_data(input_path)
    
    # Run pipeline
    pipeline = DataPipeline(spark_session)
    pipeline.run(input_path, output_path)
    
    # Validate output
    result_df = spark_session.read.parquet(output_path)
    assert result_df.count() > 0
    assert validate_output_schema(result_df)
```

### 3. Data Testing
Validate data quality and business rules.

```python
def test_data_quality(df):
    # Test data completeness
    assert df.count() > 0, "Dataset should not be empty"
    
    # Test data integrity
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) 
        for c in df.columns
    ]).collect()[0]
    
    critical_columns = ["user_id", "transaction_id", "amount"]
    for col in critical_columns:
        assert null_counts[col] == 0, f"Critical column {col} has null values"
    
    # Test business rules
    invalid_amounts = df.filter(F.col("amount") <= 0).count()
    assert invalid_amounts == 0, "All amounts should be positive"
```

## Error Handling and Reliability

### 1. Robust Error Handling
Implement comprehensive error handling with proper logging and recovery mechanisms.

```python
import logging
from typing import Optional
from functools import wraps

def retry_on_failure(max_retries: int = 3, delay: int = 1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logging.error(f"All {max_retries} attempts failed")
                        raise
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        return wrapper
    return decorator

class DataPipelineError(Exception):
    """Custom exception for data pipeline errors"""
    def __init__(self, message: str, error_code: str, context: Dict[str, Any]):
        self.message = message
        self.error_code = error_code
        self.context = context
        super().__init__(self.message)

@retry_on_failure(max_retries=3)
def process_data_with_error_handling(df):
    try:
        # Data processing logic
        result = df.transform(complex_transformation)
        
        # Validate result
        if result.count() == 0:
            raise DataPipelineError(
                message="Transformation resulted in empty dataset",
                error_code="EMPTY_RESULT",
                context={"input_count": df.count()}
            )
        
        return result
        
    except Exception as e:
        logging.error(f"Data processing failed: {e}")
        # Send to dead letter queue or error handling system
        handle_failed_batch(df, str(e))
        raise
```

### 2. Circuit Breaker Pattern
Implement circuit breakers for external dependencies.

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        return (datetime.now() - self.last_failure_time).seconds >= self.timeout
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

## Performance Optimization

### 1. Resource Management
- Right-size compute resources based on workload characteristics
- Use resource pools and queues for workload isolation
- Implement auto-scaling for variable workloads
- Monitor resource utilization and costs

### 2. Data Partitioning Strategies
```python
# Time-based partitioning
df.write \
  .partitionBy("year", "month", "day") \
  .parquet("time_partitioned_data")

# Key-based partitioning with salt for even distribution
from pyspark.sql.functions import hash, abs as abs_func

df_with_salt = df.withColumn(
    "partition_key", 
    abs_func(hash(col("user_id")) % 100)
)

df_with_salt.write \
  .partitionBy("partition_key") \
  .parquet("hash_partitioned_data")
```

### 3. Caching and Materialization
- Cache intermediate results for reuse
- Use appropriate storage levels based on memory constraints
- Materialize frequently accessed derived datasets
- Implement cache warming strategies

## Monitoring and Observability

### 1. Comprehensive Logging
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_pipeline_start(self, pipeline_name: str, config: Dict[str, Any]):
        self.logger.info(json.dumps({
            "event": "pipeline_start",
            "pipeline_name": pipeline_name,
            "timestamp": datetime.now().isoformat(),
            "config": config
        }))
    
    def log_stage_completion(self, stage_name: str, metrics: Dict[str, Any]):
        self.logger.info(json.dumps({
            "event": "stage_complete",
            "stage_name": stage_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }))
```

### 2. Metrics and Alerting
- Track pipeline SLAs and performance metrics
- Implement data freshness monitoring
- Set up alerting for failures and anomalies
- Monitor data volume and quality trends

### 3. Health Checks
```python
def perform_health_check() -> Dict[str, Any]:
    checks = {
        "database_connectivity": check_database_connection(),
        "data_freshness": check_data_freshness(),
        "pipeline_status": check_recent_pipeline_runs(),
        "resource_usage": check_resource_utilization()
    }
    
    overall_health = all(checks.values())
    
    return {
        "healthy": overall_health,
        "timestamp": datetime.now().isoformat(),
        "checks": checks
    }
```

## Security and Compliance

### 1. Data Security
- Encrypt data at rest and in transit
- Implement proper access controls and authentication
- Use secrets management for credentials
- Audit data access and modifications

### 2. Privacy and Compliance
- Implement data masking and anonymization
- Maintain audit trails for compliance
- Implement data retention policies
- Ensure GDPR, HIPAA, or other regulatory compliance

This comprehensive guide provides the foundation for building robust, scalable, and maintainable data engineering solutions.
