# Real-Time Streaming Data Processing

## Apache Kafka

### Architecture and Core Concepts

#### Key Components
- **Broker**: Kafka server that stores and serves data
- **Topic**: Category or stream name for organizing messages
- **Partition**: Ordered, immutable sequence of messages within a topic
- **Producer**: Client that publishes messages to topics
- **Consumer**: Client that subscribes to topics and processes messages
- **Consumer Group**: Set of consumers that jointly consume messages from topics

#### Message Structure
```python
# Kafka message components
{
    "key": "user_123",           # Optional partition key
    "value": "{json_payload}",   # Message content
    "timestamp": 1642694400000,  # Message timestamp
    "headers": {                 # Optional metadata
        "source": "web_app",
        "version": "1.0"
    }
}
```

### Kafka Configuration Best Practices

#### Producer Configuration
```python
from kafka import KafkaProducer
import json

producer_config = {
    'bootstrap_servers': ['kafka1:9092', 'kafka2:9092', 'kafka3:9092'],
    'acks': 'all',                    # Wait for all replicas to acknowledge
    'retries': 3,                     # Retry failed sends
    'batch_size': 16384,             # Batch size in bytes
    'linger_ms': 5,                  # Wait time for batching
    'buffer_memory': 33554432,       # Producer buffer memory
    'compression_type': 'snappy',    # Compression algorithm
    'enable_idempotence': True,      # Prevent duplicate messages
    'max_in_flight_requests_per_connection': 5,
    'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
    'key_serializer': lambda x: x.encode('utf-8') if x else None
}

producer = KafkaProducer(**producer_config)

def send_message(topic: str, key: str, value: dict):
    try:
        future = producer.send(topic, key=key, value=value)
        record_metadata = future.get(timeout=10)
        print(f"Message sent to {record_metadata.topic} partition {record_metadata.partition}")
    except Exception as e:
        print(f"Failed to send message: {e}")

# Example usage
send_message('user_events', 'user_123', {
    'user_id': 'user_123',
    'event_type': 'page_view',
    'page': '/products',
    'timestamp': '2023-01-01T10:00:00Z'
})
```

#### Consumer Configuration
```python
from kafka import KafkaConsumer
import json

consumer_config = {
    'bootstrap_servers': ['kafka1:9092', 'kafka2:9092', 'kafka3:9092'],
    'group_id': 'analytics_consumer_group',
    'auto_offset_reset': 'earliest',     # Start from beginning if no offset
    'enable_auto_commit': False,         # Manual offset management
    'max_poll_records': 500,            # Max records per poll
    'session_timeout_ms': 30000,        # Session timeout
    'heartbeat_interval_ms': 3000,      # Heartbeat interval
    'fetch_min_bytes': 1024,            # Minimum fetch size
    'fetch_max_wait_ms': 500,           # Maximum wait time for fetch
    'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
    'key_deserializer': lambda x: x.decode('utf-8') if x else None
}

consumer = KafkaConsumer('user_events', **consumer_config)

def consume_messages():
    try:
        for message in consumer:
            try:
                # Process message
                process_event(message.value)
                
                # Commit offset after successful processing
                consumer.commit()
                
            except Exception as e:
                print(f"Error processing message: {e}")
                # Handle failed message (dead letter queue, retry, etc.)
                
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        consumer.close()

def process_event(event_data: dict):
    """Process individual event"""
    print(f"Processing event: {event_data}")
    # Your business logic here
```

### Kafka Streams Processing

#### Stream Processing Application
```python
# Using kafka-python for stream processing
from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict, deque
import json
import time
from datetime import datetime, timedelta

class StreamProcessor:
    def __init__(self, input_topic: str, output_topic: str):
        self.input_topic = input_topic
        self.output_topic = output_topic
        
        # Consumer for input stream
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=['localhost:9092'],
            group_id='stream_processor',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Producer for output stream
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        # Windowed aggregation state
        self.window_size = timedelta(minutes=5)
        self.windows = defaultdict(lambda: defaultdict(int))
    
    def process_stream(self):
        """Process streaming data with windowed aggregations"""
        for message in self.consumer:
            try:
                event = message.value
                self.process_event(event)
                
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def process_event(self, event: dict):
        """Process individual event with windowing"""
        timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
        user_id = event['user_id']
        event_type = event['event_type']
        
        # Create window key (5-minute tumbling window)
        window_start = timestamp.replace(
            minute=(timestamp.minute // 5) * 5,
            second=0,
            microsecond=0
        )
        
        window_key = window_start.isoformat()
        
        # Update window aggregation
        self.windows[window_key][f"{user_id}_{event_type}"] += 1
        
        # Check if window is complete (simple time-based trigger)
        if self.is_window_complete(window_start):
            self.emit_window_result(window_key)
    
    def is_window_complete(self, window_start: datetime) -> bool:
        """Check if window should be emitted"""
        now = datetime.now(window_start.tzinfo)
        return now >= window_start + self.window_size + timedelta(minutes=1)  # 1 min grace period
    
    def emit_window_result(self, window_key: str):
        """Emit aggregated results for completed window"""
        window_data = self.windows[window_key]
        
        result = {
            'window_start': window_key,
            'window_end': (datetime.fromisoformat(window_key) + self.window_size).isoformat(),
            'aggregations': dict(window_data),
            'total_events': sum(window_data.values())
        }
        
        # Send to output topic
        self.producer.send(self.output_topic, value=result)
        
        # Clean up processed window
        del self.windows[window_key]
        
        print(f"Emitted window result: {result}")

# Usage
processor = StreamProcessor('user_events', 'user_aggregations')
processor.process_stream()
```

## Apache Flink

### Core Concepts
- **DataStream API**: For stream processing
- **DataSet API**: For batch processing (deprecated in favor of unified DataStream)
- **Table API & SQL**: High-level declarative APIs
- **Checkpointing**: Fault tolerance mechanism
- **Watermarks**: Handle event time and late data

### Flink Stream Processing Example
```python
# PyFlink example for stream processing
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.table.descriptors import Kafka, Json, Schema
from pyflink.table.table_descriptor import TableDescriptor
from pyflink.table.expressions import col

def create_kafka_source_table(table_env):
    """Create Kafka source table"""
    table_env.create_temporary_table(
        'user_events',
        TableDescriptor.for_connector('kafka')
        .schema(Schema.new_builder()
                .column('user_id', 'STRING')
                .column('event_type', 'STRING')
                .column('timestamp_field', 'TIMESTAMP(3)')
                .watermark('timestamp_field', 'timestamp_field - INTERVAL \'5\' SECOND')
                .build())
        .option('topic', 'user_events')
        .option('properties.bootstrap.servers', 'localhost:9092')
        .option('properties.group.id', 'flink_consumer')
        .option('scan.startup.mode', 'latest-offset')
        .format('json')
        .build()
    )

def create_processing_pipeline():
    """Create Flink processing pipeline"""
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)
    env.enable_checkpointing(60000)  # Checkpoint every minute
    
    table_env = StreamTableEnvironment.create(env)
    
    # Create source table
    create_kafka_source_table(table_env)
    
    # Process data with SQL
    result_table = table_env.sql_query("""
        SELECT 
            user_id,
            event_type,
            TUMBLE_START(timestamp_field, INTERVAL '5' MINUTE) as window_start,
            TUMBLE_END(timestamp_field, INTERVAL '5' MINUTE) as window_end,
            COUNT(*) as event_count
        FROM user_events
        GROUP BY 
            user_id, 
            event_type,
            TUMBLE(timestamp_field, INTERVAL '5' MINUTE)
    """)
    
    # Create sink table
    table_env.create_temporary_table(
        'user_aggregations',
        TableDescriptor.for_connector('kafka')
        .schema(Schema.new_builder()
                .column('user_id', 'STRING')
                .column('event_type', 'STRING')
                .column('window_start', 'TIMESTAMP(3)')
                .column('window_end', 'TIMESTAMP(3)')
                .column('event_count', 'BIGINT')
                .build())
        .option('topic', 'user_aggregations')
        .option('properties.bootstrap.servers', 'localhost:9092')
        .format('json')
        .build()
    )
    
    # Insert results into sink
    result_table.execute_insert('user_aggregations')

# Execute pipeline
create_processing_pipeline()
```

## Apache Storm

### Topology Architecture
- **Spout**: Source of streams (data ingestion)
- **Bolt**: Processing logic (transformation, aggregation)
- **Tuple**: Unit of data flowing through topology
- **Stream**: Unbounded sequence of tuples

### Storm Topology Example
```python
# Using streamparse for Python Storm development
from streamparse import Spout, Bolt, Topology
import json
import random

class EventSpout(Spout):
    """Generates sample events"""
    
    def next_tuple(self):
        event = {
            'user_id': f'user_{random.randint(1, 1000)}',
            'event_type': random.choice(['click', 'view', 'purchase']),
            'timestamp': int(time.time() * 1000),
            'value': random.randint(1, 100)
        }
        
        self.emit([json.dumps(event)])

class ParseEventBolt(Bolt):
    """Parse and validate events"""
    
    def process(self, tup):
        try:
            event_json = tup.values[0]
            event = json.loads(event_json)
            
            # Validate event
            if self.is_valid_event(event):
                self.emit([event])
            else:
                self.logger.warning(f"Invalid event: {event}")
                
        except Exception as e:
            self.logger.error(f"Error parsing event: {e}")
    
    def is_valid_event(self, event):
        required_fields = ['user_id', 'event_type', 'timestamp']
        return all(field in event for field in required_fields)

class AggregationBolt(Bolt):
    """Aggregate events by user and type"""
    
    def initialize(self, conf, context):
        self.aggregations = {}
        self.window_size = 60  # 1 minute window
    
    def process(self, tup):
        event = tup.values[0]
        user_id = event['user_id']
        event_type = event['event_type']
        timestamp = event['timestamp']
        
        # Create window key
        window = timestamp // (self.window_size * 1000) * (self.window_size * 1000)
        key = f"{user_id}_{event_type}_{window}"
        
        # Update aggregation
        if key not in self.aggregations:
            self.aggregations[key] = {
                'user_id': user_id,
                'event_type': event_type,
                'window_start': window,
                'count': 0,
                'total_value': 0
            }
        
        self.aggregations[key]['count'] += 1
        self.aggregations[key]['total_value'] += event.get('value', 0)
        
        # Emit aggregation (could implement windowing logic here)
        self.emit([self.aggregations[key]])

class EventProcessingTopology(Topology):
    """Define topology structure"""
    
    event_spout = EventSpout.spec(par=1)
    parse_bolt = ParseEventBolt.spec(inputs=[event_spout], par=2)
    agg_bolt = AggregationBolt.spec(inputs=[parse_bolt], par=4)
```

## Stream Processing Best Practices

### 1. Event Time vs Processing Time
```python
# Handle event time correctly
class EventTimeProcessor:
    def __init__(self, watermark_delay_ms: int = 5000):
        self.watermark_delay = watermark_delay_ms
        self.current_watermark = 0
    
    def process_event(self, event):
        event_time = event['timestamp']
        
        # Update watermark (simplified)
        potential_watermark = event_time - self.watermark_delay
        if potential_watermark > self.current_watermark:
            self.current_watermark = potential_watermark
            self.emit_watermark(self.current_watermark)
        
        # Process event if not too late
        if event_time >= self.current_watermark:
            return self.handle_event(event)
        else:
            return self.handle_late_event(event)
    
    def handle_event(self, event):
        """Handle on-time event"""
        pass
    
    def handle_late_event(self, event):
        """Handle late event"""
        # Could send to side output, ignore, or update with late data flag
        pass
```

### 2. Backpressure Handling
```python
class BackpressureHandler:
    def __init__(self, max_queue_size: int = 1000):
        self.queue = []
        self.max_queue_size = max_queue_size
        self.is_backpressure_enabled = False
    
    def handle_message(self, message):
        if len(self.queue) >= self.max_queue_size:
            if not self.is_backpressure_enabled:
                self.enable_backpressure()
            return False  # Drop message or apply backpressure
        
        self.queue.append(message)
        self.process_message(message)
        
        if self.is_backpressure_enabled and len(self.queue) < self.max_queue_size * 0.7:
            self.disable_backpressure()
        
        return True
    
    def enable_backpressure(self):
        self.is_backpressure_enabled = True
        print("Backpressure enabled")
    
    def disable_backpressure(self):
        self.is_backpressure_enabled = False
        print("Backpressure disabled")
```

### 3. Exactly-Once Processing
```python
class ExactlyOnceProcessor:
    def __init__(self, checkpoint_interval: int = 60):
        self.processed_offsets = {}
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = time.time()
    
    def process_message(self, topic: str, partition: int, offset: int, message):
        key = f"{topic}_{partition}"
        
        # Check if already processed
        if key in self.processed_offsets and offset <= self.processed_offsets[key]:
            print(f"Message already processed: {topic}:{partition}:{offset}")
            return
        
        try:
            # Process message
            result = self.handle_message(message)
            
            # Update processed offset
            self.processed_offsets[key] = offset
            
            # Checkpoint if needed
            if time.time() - self.last_checkpoint > self.checkpoint_interval:
                self.checkpoint()
            
            return result
            
        except Exception as e:
            print(f"Error processing message: {e}")
            # Don't update offset on failure
            raise
    
    def checkpoint(self):
        """Save current state for recovery"""
        self.save_state(self.processed_offsets)
        self.last_checkpoint = time.time()
        print("Checkpoint saved")
```

### 4. Monitoring and Alerting
```python
import time
from collections import defaultdict

class StreamingMetrics:
    def __init__(self):
        self.metrics = defaultdict(int)
        self.latency_samples = []
        self.start_time = time.time()
    
    def record_message_processed(self):
        self.metrics['messages_processed'] += 1
    
    def record_message_failed(self):
        self.metrics['messages_failed'] += 1
    
    def record_latency(self, latency_ms: float):
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]  # Keep last 1000 samples
    
    def get_processing_rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.metrics['messages_processed'] / elapsed if elapsed > 0 else 0
    
    def get_error_rate(self) -> float:
        total = self.metrics['messages_processed'] + self.metrics['messages_failed']
        return self.metrics['messages_failed'] / total if total > 0 else 0
    
    def get_average_latency(self) -> float:
        return sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
    
    def check_health(self) -> dict:
        return {
            'processing_rate': self.get_processing_rate(),
            'error_rate': self.get_error_rate(),
            'average_latency_ms': self.get_average_latency(),
            'total_processed': self.metrics['messages_processed'],
            'total_failed': self.metrics['messages_failed']
        }
```

This comprehensive guide covers the essential aspects of real-time streaming data processing with practical examples and best practices for building robust streaming applications.
