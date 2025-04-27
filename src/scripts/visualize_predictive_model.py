import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def create_visualization_df(results, metric_names):
    """Create a lightweight dataframe optimized for visualization."""
    models = list(results.keys())
    viz_data = {}
    
    for metric in metric_names:
        metric_values = [results[model][metric] for model in models]
        viz_data[metric] = metric_values
        
    return pd.DataFrame(viz_data, index=models)

def plot_confusion_matrices(models, X_test, y_test, target_name):
    """Plot an enhanced confusion matrix with performance metrics."""
    # Get predictions from best model (assuming it's the last model)
    best_model = list(models.values())[-1]
    y_pred = best_model.predict(X_test)
    
    # Get class labels
    class_labels = np.unique(y_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate metrics - create a small temporary dataframe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=class_labels
    )
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create metrics dataframe - only store what's needed for visualization
    metrics_df = pd.DataFrame({
        'Class': [str(label) for label in class_labels] + ['Weighted Avg', 'Accuracy'],
        'Precision': list(precision) + [np.mean(precision, weights=support), accuracy],
        'Recall': list(recall) + [np.mean(recall, weights=support), ''],
        'F1': list(f1) + [np.mean(f1, weights=support), ''],
        'Support': list(support) + [sum(support), '']
    })
    
    # Create a subplot with 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=["Confusion Matrix", "Performance Metrics"],
        specs=[[{"type": "heatmap"}, {"type": "table"}]]
    )
    
    # Add confusion matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            colorscale='Blues',
            showscale=False,
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
            name=''
        ),
        row=1, col=1
    )
    
    # Add count annotations to the heatmap
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            fig.add_annotation(
                text=str(cm[i, j]),
                x=class_labels[j],
                y=class_labels[i],
                showarrow=False,
                font=dict(
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    size=14
                ),
                row=1, col=1
            )
    
    # Add metrics table using the metrics dataframe
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(metrics_df.columns),
                fill_color='royalblue',
                align='center',
                font=dict(color='white', size=14),
                height=30
            ),
            cells=dict(
                values=[metrics_df[col] for col in metrics_df.columns],
                fill_color=[
                    ['white'] * (len(metrics_df) - 2) + ['rgba(211, 211, 211, 0.3)'] * 2,
                    'white'
                ],
                align='center',
                height=25,
                font=dict(size=13),
                format=[None, '.4f', '.4f', '.4f', None]
            )
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Classification Performance for {target_name}",
            x=0.5,
            font=dict(size=18)
        ),
        height=500,
        width=1000,
        template='plotly_white',
        showlegend=False,
    )
    
    # Update axes
    fig.update_xaxes(title_text="Predicted", title_font=dict(size=14), row=1, col=1)
    fig.update_yaxes(title_text="Actual", title_font=dict(size=14), row=1, col=1)
    
    return fig

def plot_feature_importance(model, X_train, target_name, categorical_cols, numerical_cols, top_n=15):
    """Plot feature importance with a lightweight visualization approach."""
    # Check if model has feature importances attribute
    if not hasattr(model.named_steps['model'], 'feature_importances_'):
        print(f"Model doesn't provide feature importances")
        return
    
    # Get feature importances
    importances = model.named_steps['model'].feature_importances_
    print(f"Number of importance values: {len(importances)}")
    
    # Get the preprocessing transformer
    preprocessor = model.named_steps['preprocessor']
    
    # Initialize the combined feature list
    feature_names = []
    
    # Process categorical features if any exist
    if categorical_cols:
        print(f"Processing categorical columns: {categorical_cols}")
        # Find the categorical transformer
        cat_transformer = None
        cat_cols = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'cat' and any(col in cols for col in categorical_cols):
                cat_transformer = transformer
                cat_cols = cols
                print(f"Found categorical transformer with columns: {cols}")
                break
        
        # If categorical transformer is found
        if cat_transformer is not None:
            # Try to get OneHotEncoder
            onehot_step = None
            
            # First check if transformer is itself an encoder (not in a pipeline)
            if isinstance(cat_transformer, OneHotEncoder):
                onehot_step = cat_transformer
                print("Categorical transformer is directly an OneHotEncoder")
            else:
                # Otherwise check within pipeline steps
                try:
                    for step_name, step in cat_transformer.named_steps.items():
                        if isinstance(step, OneHotEncoder):
                            onehot_step = step
                            print(f"Found OneHotEncoder in step: {step_name}")
                            break
                except AttributeError:
                    print("Categorical transformer doesn't have named_steps attribute")
            
            if onehot_step:
                try:
                    print("Categories shape:", [len(cats) for cats in onehot_step.categories_])
                    # Get proper category names from each categorical column
                    for i, col in enumerate(cat_cols):
                        if col in categorical_cols:
                            print(f"Processing {col} at position {i}")
                            # Get categories for this column
                            categories = onehot_step.categories_[i]
                            print(f"Found {len(categories)} categories for {col}: {categories[:5]}")
                            # Create feature names like "pnns_groups_1=beverages"
                            for category in categories:
                                feature_names.append(f"{col}={category}")
                except Exception as e:
                    print(f"Error getting categorical feature names: {e}")
                    # Fallback: use basic column names
                    feature_names.extend(categorical_cols)
            else:
                print("No OneHotEncoder found in categorical transformer")
                feature_names.extend(categorical_cols)
        else:
            print("No categorical transformer found")
            feature_names.extend(categorical_cols)
    
    # Add numerical feature names
    feature_names.extend(numerical_cols)
    print(f"Total feature names collected: {len(feature_names)}")
    
    # Ensure we have the right number of feature names
    if len(feature_names) > len(importances):
        print(f"WARNING: More feature names ({len(feature_names)}) than importances ({len(importances)}). Trimming feature names.")
        feature_names = feature_names[:len(importances)]
    elif len(feature_names) < len(importances):
        print(f"WARNING: Fewer feature names ({len(feature_names)}) than importances ({len(importances)}). Adding generic names.")
        # Try to make more meaningful names for extra features
        for i in range(len(feature_names), len(importances)):
            feature_names.append(f"Feature_{i}")
    
    # Create small DataFrame with only top features for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort and get top features
    top_features = feature_importance.sort_values(
        by='Importance', ascending=False
    ).head(top_n)
    
    # Create enhanced bar plot with reduced data
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(
            color=top_features['Importance'],
            colorscale='RdBu_r',
            colorbar=dict(title='Importance'),
            line=dict(width=1, color='black')
        )
    ))
    
    # Update layout with more professional styling
    fig.update_layout(
        title=dict(
            text=f'Top {top_n} Features for Predicting {target_name}',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(
                text='Feature Importance',
                font=dict(size=14)
            ),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(
                text='Feature',
                font=dict(size=14)
            ),
            categoryorder='total ascending'
        ),
        template='plotly_white',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def plot_regression_results(results, target_name):
    """Create a lightweight bar chart to compare regression model performance."""
    # Create a tiny dataframe just for visualization
    viz_df = create_visualization_df(results, ['RMSE', 'R²'])
    models = viz_df.index.tolist()
    
    fig = go.Figure()
    
    # Add RMSE bars
    fig.add_trace(go.Bar(
        x=models,
        y=viz_df['RMSE'],
        name='RMSE',
        marker_color='indianred'
    ))
    
    # Add R² on secondary axis
    fig.add_trace(go.Scatter(
        x=models,
        y=viz_df['R²'],
        name='R²',
        mode='markers',
        marker=dict(size=12, color='royalblue'),
        yaxis='y2'
    ))
    
    # Layout with two y-axes
    fig.update_layout(
        title=f'Model Performance Comparison for {target_name}',
        yaxis=dict(
            title='RMSE (lower is better)',
            side='left'
        ),
        yaxis2=dict(
            title='R² (higher is better)',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    return fig

def plot_classification_results(results, target_name):
    """Create a lightweight bar chart to compare classification model performance."""
    # Create a tiny dataframe just for visualization
    viz_df = create_visualization_df(results, ['Accuracy', 'F1 Score'])
    models = viz_df.index.tolist()
    
    fig = go.Figure()
    
    # Add accuracy bars
    fig.add_trace(go.Bar(
        x=models,
        y=viz_df['Accuracy'],
        name='Accuracy',
        marker_color='forestgreen'
    ))
    
    # Add F1 score bars
    fig.add_trace(go.Bar(
        x=models,
        y=viz_df['F1 Score'],
        name='F1 Score',
        marker_color='darkorange'
    ))
    
    # Layout
    fig.update_layout(
        title=f'Model Performance Comparison for {target_name}',
        yaxis=dict(
            title='Score (higher is better)',
            range=[0, 1]
        ),
        barmode='group',
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    return fig