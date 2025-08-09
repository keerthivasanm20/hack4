# Cloud Data Platforms Guide

## Amazon Web Services (AWS)

### Core Data Services

#### Amazon S3 (Simple Storage Service)
- **Use Cases**: Data lakes, backup, archival, content distribution
- **Storage Classes**: Standard, IA, One Zone-IA, Glacier, Deep Archive
- **Best Practices**:
  - Use lifecycle policies for cost optimization
  - Enable versioning for critical data
  - Implement proper IAM policies and bucket policies
  - Use S3 Select for query-in-place capabilities

```python
# S3 best practices example
import boto3
from botocore.exceptions import ClientError

class S3Manager:
    def __init__(self, region_name='us-east-1'):
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.s3_resource = boto3.resource('s3', region_name=region_name)
    
    def upload_with_metadata(self, bucket: str, key: str, file_path: str, metadata: dict):
        try:
            self.s3_client.upload_file(
                file_path, 
                bucket, 
                key,
                ExtraArgs={
                    'Metadata': metadata,
                    'ServerSideEncryption': 'AES256'
                }
            )
        except ClientError as e:
            print(f"Upload failed: {e}")
    
    def create_lifecycle_policy(self, bucket: str):
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'data-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'data/'},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }
        
        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket,
            LifecycleConfiguration=lifecycle_config
        )
```

#### Amazon Redshift
- **Use Cases**: Data warehousing, analytics, BI reporting
- **Features**: Columnar storage, parallel processing, auto-scaling
- **Optimization Tips**:
  - Choose appropriate distribution keys
  - Use sort keys for query performance
  - Implement proper compression
  - Regular VACUUM and ANALYZE operations

#### AWS Glue
- **Components**: Data Catalog, ETL Jobs, Crawlers, Workflows
- **Features**: Serverless, auto-scaling, schema discovery
- **Best Practices**:
  - Use Glue Data Catalog as central metadata repository
  - Implement incremental processing patterns
  - Use job bookmarks for state management

#### Amazon EMR (Elastic MapReduce)
- **Use Cases**: Big data processing with Spark, Hadoop, Hive
- **Cost Optimization**:
  - Use Spot instances for cost savings
  - Implement auto-scaling groups
  - Choose appropriate instance types

#### Amazon Kinesis
- **Components**: Data Streams, Data Firehose, Analytics
- **Use Cases**: Real-time data ingestion and processing
- **Scaling**: Shard-based scaling for throughput requirements

### AWS Data Pipeline Architecture Example
```python
# Complete AWS data pipeline using boto3
import boto3
import json
from datetime import datetime

class AWSDataPipeline:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.glue = boto3.client('glue')
        self.kinesis = boto3.client('kinesis')
        self.redshift = boto3.client('redshift-data')
    
    def setup_data_lake(self, bucket_name: str):
        """Setup S3 data lake structure"""
        folders = [
            'raw-data/',
            'processed-data/',
            'curated-data/',
            'temp/',
            'logs/'
        ]
        
        for folder in folders:
            self.s3.put_object(Bucket=bucket_name, Key=folder)
    
    def create_glue_job(self, job_name: str, script_location: str):
        """Create Glue ETL job"""
        response = self.glue.create_job(
            Name=job_name,
            Role='AWSGlueServiceRole',
            Command={
                'Name': 'glueetl',
                'ScriptLocation': script_location,
                'PythonVersion': '3'
            },
            DefaultArguments={
                '--TempDir': 's3://my-bucket/temp/',
                '--job-bookmark-option': 'job-bookmark-enable'
            },
            MaxRetries=3,
            GlueVersion='3.0'
        )
        return response
    
    def run_pipeline(self, source_bucket: str, target_table: str):
        """Execute complete pipeline"""
        try:
            # 1. Trigger Glue crawler
            self.glue.start_crawler(Name='data-crawler')
            
            # 2. Run ETL job
            self.glue.start_job_run(JobName='etl-job')
            
            # 3. Load to Redshift
            copy_command = f"""
            COPY {target_table}
            FROM 's3://{source_bucket}/processed-data/'
            IAM_ROLE 'arn:aws:iam::account:role/RedshiftRole'
            FORMAT AS PARQUET;
            """
            
            self.redshift.execute_statement(
                ClusterIdentifier='my-redshift-cluster',
                Database='dev',
                Sql=copy_command
            )
            
            print("Pipeline executed successfully")
            
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
```

## Google Cloud Platform (GCP)

### Core Data Services

#### Google Cloud Storage
- **Use Cases**: Data lakes, backup, serving web content
- **Storage Classes**: Standard, Nearline, Coldline, Archive
- **Features**: Global availability, strong consistency, lifecycle management

#### BigQuery
- **Architecture**: Serverless, columnar, massively parallel
- **Features**: Standard SQL, ML integration, real-time analytics
- **Best Practices**:
  - Partition tables by date for cost optimization
  - Use clustering for better query performance
  - Implement proper slot management

```sql
-- BigQuery optimization examples

-- Create partitioned and clustered table
CREATE TABLE `project.dataset.sales_data`
(
  sale_id STRING,
  customer_id STRING,
  product_id STRING,
  sale_date DATE,
  amount NUMERIC
)
PARTITION BY sale_date
CLUSTER BY customer_id, product_id;

-- Use partition pruning in queries
SELECT customer_id, SUM(amount) as total_sales
FROM `project.dataset.sales_data`
WHERE sale_date BETWEEN '2023-01-01' AND '2023-12-31'
  AND customer_id IN ('cust1', 'cust2')
GROUP BY customer_id;

-- Optimize with approximate aggregation for large datasets
SELECT 
  APPROX_COUNT_DISTINCT(customer_id) as unique_customers,
  APPROX_QUANTILES(amount, 4) as amount_quartiles
FROM `project.dataset.sales_data`
WHERE sale_date >= '2023-01-01';
```

#### Cloud Dataflow
- **Based on**: Apache Beam
- **Features**: Unified batch and stream processing
- **Auto-scaling**: Dynamic worker allocation

#### Cloud Pub/Sub
- **Use Cases**: Event-driven architectures, real-time messaging
- **Features**: At-least-once delivery, global availability
- **Integration**: Native integration with Dataflow and other GCP services

#### Cloud Dataproc
- **Use Cases**: Managed Spark and Hadoop clusters
- **Features**: Fast cluster creation, preemptible instances
- **Cost Optimization**: Use preemptible VMs for batch workloads

### GCP Data Pipeline Example
```python
# GCP data pipeline using Cloud SDK
from google.cloud import storage, bigquery, pubsub_v1
from google.cloud import dataflow_v1beta3

class GCPDataPipeline:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.storage_client = storage.Client()
        self.bigquery_client = bigquery.Client()
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
    
    def create_bigquery_dataset(self, dataset_id: str):
        """Create BigQuery dataset"""
        dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
        dataset.location = "US"
        
        dataset = self.bigquery_client.create_dataset(dataset, exists_ok=True)
        print(f"Created dataset {dataset.dataset_id}")
    
    def load_data_to_bigquery(self, dataset_id: str, table_id: str, source_uri: str):
        """Load data from GCS to BigQuery"""
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        
        load_job = self.bigquery_client.load_table_from_uri(
            source_uri, table_ref, job_config=job_config
        )
        
        load_job.result()  # Wait for job completion
        print(f"Loaded {load_job.output_rows} rows to {table_ref}")
    
    def publish_message(self, topic_name: str, message: str):
        """Publish message to Pub/Sub"""
        topic_path = self.publisher.topic_path(self.project_id, topic_name)
        
        data = message.encode('utf-8')
        future = self.publisher.publish(topic_path, data)
        
        message_id = future.result()
        print(f"Published message {message_id}")
```

## Microsoft Azure

### Core Data Services

#### Azure Data Lake Storage Gen2
- **Features**: Hierarchical namespace, fine-grained access control
- **Integration**: Native integration with Azure analytics services
- **Security**: Azure Active Directory integration, encryption

#### Azure Synapse Analytics
- **Components**: SQL pools, Spark pools, Data Explorer pools
- **Features**: Unified analytics, serverless and dedicated options
- **Use Cases**: Data warehousing, big data analytics, real-time analytics

#### Azure Data Factory
- **Features**: Visual ETL/ELT design, hybrid data integration
- **Components**: Pipelines, Activities, Datasets, Linked Services
- **Monitoring**: Built-in monitoring and alerting capabilities

```json
// Azure Data Factory pipeline example
{
    "name": "DataProcessingPipeline",
    "properties": {
        "activities": [
            {
                "name": "CopyDataActivity",
                "type": "Copy",
                "inputs": [
                    {
                        "referenceName": "SourceDataset",
                        "type": "DatasetReference"
                    }
                ],
                "outputs": [
                    {
                        "referenceName": "DestinationDataset",
                        "type": "DatasetReference"
                    }
                ],
                "typeProperties": {
                    "source": {
                        "type": "BlobSource"
                    },
                    "sink": {
                        "type": "SqlDWSink",
                        "writeBatchSize": 10000
                    }
                }
            },
            {
                "name": "DataTransformation",
                "type": "HDInsightSpark",
                "dependsOn": [
                    {
                        "activity": "CopyDataActivity",
                        "dependencyConditions": ["Succeeded"]
                    }
                ],
                "typeProperties": {
                    "rootPath": "scripts/",
                    "entryFilePath": "transform_data.py",
                    "sparkJobLinkedService": {
                        "referenceName": "SparkLinkedService",
                        "type": "LinkedServiceReference"
                    }
                }
            }
        ]
    }
}
```

#### Azure Stream Analytics
- **Use Cases**: Real-time stream processing
- **Query Language**: SQL-like syntax for stream processing
- **Integration**: Input from Event Hubs, IoT Hub; Output to various sinks

#### Azure Event Hubs
- **Features**: Big data streaming platform
- **Capabilities**: Event ingestion, partitioning, consumer groups
- **Integration**: Native integration with Azure services

### Azure Data Pipeline Example
```python
# Azure data pipeline using Azure SDK
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import pyodbc

class AzureDataPipeline:
    def __init__(self, storage_account: str, container: str):
        self.credential = DefaultAzureCredential()
        self.blob_service = BlobServiceClient(
            account_url=f"https://{storage_account}.blob.core.windows.net",
            credential=self.credential
        )
        self.container = container
    
    def upload_to_blob(self, local_file: str, blob_name: str):
        """Upload file to Azure Blob Storage"""
        blob_client = self.blob_service.get_blob_client(
            container=self.container, 
            blob=blob_name
        )
        
        with open(local_file, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"Uploaded {local_file} to {blob_name}")
    
    def copy_to_synapse(self, blob_name: str, table_name: str):
        """Copy data from blob to Synapse SQL pool"""
        connection_string = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=your-synapse-workspace.sql.azuresynapse.net;"
            "Database=your-database;"
            "Authentication=ActiveDirectoryInteractive;"
        )
        
        copy_sql = f"""
        COPY INTO {table_name}
        FROM 'https://youraccount.blob.core.windows.net/{self.container}/{blob_name}'
        WITH (
            FILE_TYPE = 'PARQUET',
            CREDENTIAL = (IDENTITY = 'Managed Identity')
        )
        """
        
        with pyodbc.connect(connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute(copy_sql)
            conn.commit()
        
        print(f"Data copied to {table_name}")
```

## Multi-Cloud Considerations

### 1. Cloud-Agnostic Architecture
- Use containerization for portability
- Implement abstraction layers for cloud services
- Use open-source tools when possible
- Design for eventual migration scenarios

### 2. Data Governance Across Clouds
- Implement unified data catalog
- Standardize security and compliance policies
- Use consistent tagging and metadata strategies
- Implement cross-cloud monitoring and alerting

### 3. Cost Optimization Strategies
- Compare pricing models across providers
- Use reserved instances for predictable workloads
- Implement automated cost monitoring and alerts
- Consider multi-cloud arbitrage opportunities

### 4. Best Practices for Cloud Selection
- Evaluate based on specific use case requirements
- Consider existing technology stack and expertise
- Assess vendor lock-in risks
- Evaluate support and SLA requirements
- Consider data residency and compliance requirements

This guide provides comprehensive coverage of major cloud data platforms and practical implementation strategies for building scalable data solutions.
