
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
import re

# Set page config
st.set_page_config(page_title="COPD Treatment-Cause Explorer", layout="wide")

# Directory with data
DATA_DIR = "/mount/src/researchnetworks/COPD_Author_Widget/data"

# Load treatment mapping
treatment_mapping = {}
mapping_file = os.path.join(DATA_DIR, "treatment_mapping.txt")
if os.path.exists(mapping_file):
    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                simplified, original = parts
                treatment_mapping[simplified] = original

# Get all available treatment files with proper names
treatment_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_basic.tsv")]
treatment_options = [treatment_mapping.get(f.replace("_basic.tsv", ""), f.replace("_basic.tsv", "")) for f in treatment_files]

# Title and header
st.title("COPD Treatment-Cause Relationship Explorer")
st.markdown("This application allows you to explore relationships between treatments and causes, along with author contributions.")

# Sidebar for selecting treatment
st.sidebar.header("Select Treatment")
selected_treatment_display = st.sidebar.selectbox("Choose a treatment:", treatment_options)

# Convert display name back to filename
selected_treatment = None
for simple, original in treatment_mapping.items():
    if original == selected_treatment_display:
        selected_treatment = simple
        break

if not selected_treatment and selected_treatment_display:
    selected_treatment = re.sub(r'[^a-zA-Z0-9_]', '_', selected_treatment_display)

# Load the selected treatment data
@st.cache_data
def load_treatment_data(treatment_name):
    basic_file = os.path.join(DATA_DIR, f"{treatment_name}_basic.tsv")
    verbose_file = os.path.join(DATA_DIR, f"{treatment_name}_verbose.tsv")

    try:
        basic_df = pd.read_csv(basic_file, sep='\t')
        if os.path.exists(verbose_file):
            verbose_df = pd.read_csv(verbose_file, sep='\t')
        else:
            verbose_df = pd.DataFrame()
        return basic_df, verbose_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# This should be outside any cached function
show_matching_details = st.checkbox("Show matching details", key="matching_details_checkbox")

@st.cache_data
def load_author_data(treatment_name, cause_name, _show_details=False):
    # Normalize treatment and cause names for better matching
    def normalize_name(name):
        return re.sub(r'[^a-zA-Z0-9 ]', ' ', name).lower().strip()

    # Function to check if two strings are similar enough
    def strings_similar(str1, str2, threshold=0.6):
        # Normalize both strings
        str1_norm = set(normalize_name(str1).split())
        str2_norm = set(normalize_name(str2).split())

        # Calculate similarity based on word overlap
        common_words = str1_norm.intersection(str2_norm)
        total_words = str1_norm.union(str2_norm)

        if not total_words:
            return False

        similarity = len(common_words) / len(total_words)
        return similarity >= threshold

    # Debug logs (will be displayed only if _show_details is True)
    logs = []
    logs.append(f"Looking for matches for treatment: {treatment_name}")
    logs.append(f"Looking for matches for cause: {cause_name}")

    # Try to get data from consolidated file
    consolidated_file = os.path.join(DATA_DIR, "treatment_cause_authors.tsv")

    if os.path.exists(consolidated_file):
        try:
            # Load the consolidated file
            df = pd.read_csv(consolidated_file, sep='	')

            if df.empty:
                logs.append("Consolidated author file exists but is empty")
                return pd.DataFrame(columns=['Author', 'PMID Count', 'PMIDs']), logs

            logs.append(f"Found {len(df)} total authors in consolidated file")

            # Find matches using string similarity
            matches = []
            for _, row in df.iterrows():
                if (strings_similar(row['Treatment'], treatment_name) and
                    strings_similar(row['Cause'], cause_name)):
                    matches.append(row)

            if matches:
                # Convert matches to DataFrame
                result_df = pd.DataFrame(matches)
                logs.append(f"Found {len(result_df)} matching authors")
                return result_df, logs
            else:
                # Try a more lenient approach
                partial_matches = []
                norm_treatment = normalize_name(treatment_name)
                norm_cause = normalize_name(cause_name)

                for _, row in df.iterrows():
                    t_match = any(word in normalize_name(row['Treatment'])
                                   for word in norm_treatment.split() if len(word) > 3)
                    c_match = any(word in normalize_name(row['Cause'])
                                   for word in norm_cause.split() if len(word) > 3)

                    if t_match and c_match:
                        partial_matches.append(row)

                if partial_matches:
                    result_df = pd.DataFrame(partial_matches)
                    logs.append(f"Found {len(result_df)} partially matching authors")
                    return result_df, logs

                logs.append("No matching authors found for this treatment-cause pair")
                return pd.DataFrame(columns=['Author', 'PMID Count', 'PMIDs']), logs

        except Exception as e:
            logs.append(f"Error processing consolidated file: {str(e)}")
            return pd.DataFrame(columns=['Author', 'PMID Count', 'PMIDs']), logs
    else:
        logs.append("Consolidated author file not found")
        return pd.DataFrame(columns=['Author', 'PMID Count', 'PMIDs']), logs

# Load consolidated author data
@st.cache_data
def load_consolidated_authors():
    consolidated_file = os.path.join(DATA_DIR, "treatment_cause_authors.tsv")
    if os.path.exists(consolidated_file):
        return pd.read_csv(consolidated_file, sep='\t')
    else:
        st.warning("Consolidated author analysis file not found")
        return pd.DataFrame(columns=['Author', 'Treatment', 'Cause', 'PMID Count', 'PMIDs'])

# Main content
if selected_treatment:
    basic_df, verbose_df = load_treatment_data(selected_treatment)

    if not basic_df.empty:
        st.header(f"Analysis for: {selected_treatment_display}")

        # Display basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Connected Causes", len(basic_df))
        with col2:
            st.metric("Average Connection", f"{basic_df['Edge Weight'].mean():.3f}")
        with col3:
            st.metric("Shared Publications", basic_df["Shared Publications"].sum())
        with col4:
            st.metric("Max Connection", f"{basic_df['Edge Weight'].max():.3f}")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Causes Overview", "Network View", "Author Analysis"])

        with tab1:
            # Split into columns
            col1, col2 = st.columns([3, 2])

            with col1:
                # Causes table with filters
                st.subheader("Connected Causes")

                # Filter and sort controls
                search_term = st.text_input("Filter causes:")

                # Apply filtering
                if search_term:
                    filtered_df = basic_df[basic_df['Cause'].str.contains(search_term, case=False)]
                else:
                    filtered_df = basic_df

                # Sort controls
                col1a, col1b = st.columns([2, 2])
                with col1a:
                    sort_col = st.selectbox("Sort by:", ["Edge Weight", "Shared Publications", "Cause"])
                with col1b:
                    sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)

                # Apply sorting
                ascending = sort_order == "Ascending"
                sorted_df = filtered_df.sort_values(sort_col, ascending=ascending)

                # Show the table
                st.dataframe(
                    sorted_df,
                    column_config={
                        "Cause": st.column_config.TextColumn("Cause"),
                        "Edge Weight": st.column_config.NumberColumn("Edge Weight", format="%.3f"),
                        "Shared Publications": st.column_config.NumberColumn("Shared Publications")
                    },
                    use_container_width=True
                )

            with col2:
                # Show connection strength chart
                st.subheader("Top Connections")

                # Sort and get top causes
                top_df = basic_df.sort_values('Edge Weight', ascending=False).head(10)

                # Create horizontal bar chart
                fig = px.bar(
                    top_df,
                    y='Cause',
                    x='Edge Weight',
                    orientation='h',
                    labels={'Cause': 'Cause', 'Edge Weight': 'Connection Strength'},
                    color='Edge Weight',
                    color_continuous_scale='blues',
                    height=400,
                )

                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                # Add publications chart
                st.subheader("Shared Publications")

                pub_df = basic_df.sort_values('Shared Publications', ascending=False).head(10)

                fig = px.bar(
                    pub_df,
                    y='Cause',
                    x='Shared Publications',
                    orientation='h',
                    labels={'Cause': 'Cause', 'Shared Publications': 'Publications'},
                    color='Shared Publications',
                    color_continuous_scale='greens',
                    height=400,
                )

                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Relationship Network")

            # Network parameters
            max_nodes = st.slider("Max causes to display:", 5, 30, 15)

            # Create network graph
            G = nx.Graph()

            # Add treatment node
            G.add_node(selected_treatment_display, type="treatment")

            # Add top cause nodes based on connection strength
            top_causes = basic_df.sort_values('Edge Weight', ascending=False).head(max_nodes)

            for _, row in top_causes.iterrows():
                G.add_node(row['Cause'], type="cause")
                G.add_edge(
                    selected_treatment_display,
                    row['Cause'],
                    weight=row['Edge Weight'],
                    shared=row['Shared Publications']
                )

            # Create positions using a spring layout
            pos = nx.spring_layout(G, seed=42)

            # Prepare node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                if node == selected_treatment_display:
                    node_text.append(f"Treatment: {node}")
                    node_color.append('red')
                    node_size.append(20)
                else:
                    node_text.append(f"Cause: {node}")
                    node_color.append('blue')
                    node_size.append(15)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[n if len(n) < 30 else n[:27]+"..." for n in node_text],
                textposition="top center",
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line=dict(width=2)
                )
            )

            # Prepare edge trace
            edge_x = []
            edge_y = []
            edge_text = []

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                weight = G.edges[edge]['weight']
                shared = G.edges[edge]['shared']
                edge_text.extend([f"Weight: {weight:.3f}, Shared: {shared}", "", ""])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                text=edge_text,
                mode='lines')

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=600))

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Note**: You can hover over nodes and edges for details. Drag nodes to rearrange the network.")

        # In the Author Analysis tab:
        with tab3:
            st.subheader("Author Analysis")

            # Create a selection box for causes
            selected_cause = st.selectbox("Choose a cause to analyze authors:", basic_df['Cause'])

            if selected_cause:
                # Debug info section (outside the cached function)
                show_debug = st.checkbox("Show technical details", key="debug_checkbox")

                # Call the cached function and unpack the tuple
                author_result = load_author_data(selected_treatment_display, selected_cause)

                # Check if the result is a tuple (new format) or DataFrame (old format)
                if isinstance(author_result, tuple):
                    author_df, debug_logs = author_result
                else:
                    # Handle old function format
                    author_df = author_result
                    debug_logs = ["No detailed logs available"]

                # Display debug logs if requested
                if show_debug:
                    with st.expander("Debug Information"):
                        for log in debug_logs:
                            st.write(log)

                        # Additional debug info
                        st.write("### Sample consolidated data:")
                        consolidated_file = os.path.join(DATA_DIR, "treatment_cause_authors.tsv")
                        if os.path.exists(consolidated_file):
                            sample_df = pd.read_csv(consolidated_file, sep='	', nrows=5)
                            st.dataframe(sample_df)
                        else:
                            st.error("Consolidated file not found")

                # Display author data
                if hasattr(author_df, 'empty') and not author_df.empty:
                    # Rest of your code to display the author data...
                    st.dataframe(author_df)
                else:
                    st.warning(f"No author data available for {selected_treatment_display} and {selected_cause}")
                    st.info("Check the global author analysis for combined data.")
    else:
        st.warning("No data available for the selected treatment")

# Consolidated author analysis section
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Global Author Analysis", False):
    st.header("Consolidated Author Analysis")

    # Load data
    consolidated_df = load_consolidated_authors()

    if not consolidated_df.empty:
        # Filters
        st.subheader("Filters")

        # Set up filtering options
        col1, col2 = st.columns(2)
        with col1:
            # Top N filter
            top_n = st.number_input("Show top authors:", min_value=10, max_value=100, value=25, step=5)

            # Minimum count filter
            min_count = int(consolidated_df['PMID Count'].min())
            max_count = int(consolidated_df['PMID Count'].max())
            min_count_filter = st.slider("Minimum publications:", min_count, max_count, min_count)

        with col2:
            # Author search
            author_search = st.text_input("Search author:")

            # Treatment/cause filters if needed
            show_filters = st.checkbox("Add treatment/cause filters")

        if show_filters:
            col1a, col2a = st.columns(2)
            with col1a:
                # Get unique treatments
                treatments = sorted(consolidated_df['Treatment'].unique())
                selected_treatments = st.multiselect("Filter by treatments:", treatments)

            with col2a:
                # Get unique causes
                causes = sorted(consolidated_df['Cause'].unique())
                selected_causes = st.multiselect("Filter by causes:", causes)
        else:
            selected_treatments = []
            selected_causes = []

        # Apply filters
        filtered_df = consolidated_df[consolidated_df['PMID Count'] >= min_count_filter]

        if author_search:
            filtered_df = filtered_df[filtered_df['Author'].str.contains(author_search, case=False)]

        if selected_treatments:
            filtered_df = filtered_df[filtered_df['Treatment'].isin(selected_treatments)]

        if selected_causes:
            filtered_df = filtered_df[filtered_df['Cause'].isin(selected_causes)]

        # Display results
        if not filtered_df.empty:
            # Show summary metrics
            st.subheader("Summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Authors", filtered_df['Author'].nunique())
            with col2:
                st.metric("Total Treatments", filtered_df['Treatment'].nunique())
            with col3:
                st.metric("Total Causes", filtered_df['Cause'].nunique())

            # Display top authors
            st.subheader("Top Authors")

            # Group by author and sum PMID counts
            author_stats = filtered_df.groupby('Author')['PMID Count'].sum().reset_index()
            author_stats = author_stats.sort_values('PMID Count', ascending=False).head(top_n)

            # Show bar chart
            fig = px.bar(
                author_stats,
                x='PMID Count',
                y='Author',
                orientation='h',
                color='PMID Count',
                color_continuous_scale='Viridis',
                height=600
            )

            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Total Publication Count",
                yaxis_title="Author"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show detailed data
            st.subheader("Detailed Author Data")
            st.dataframe(
                filtered_df,
                column_config={
                    "Author": st.column_config.TextColumn("Author"),
                    "Treatment": st.column_config.TextColumn("Treatment"),
                    "Cause": st.column_config.TextColumn("Cause"),
                    "PMID Count": st.column_config.NumberColumn("PMID Count"),
                    "PMIDs": st.column_config.TextColumn("PMIDs")
                },
                use_container_width=True
            )
        else:
            st.info("No data found matching your filters")
    else:
        st.info("Consolidated author analysis file not found")

# Add a footer with instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Select a treatment from the dropdown menu
2. View connected causes and their strength
3. Explore different tabs for detailed views
4. Check the author analysis for specific treatment-cause pairs
5. Enable the global analysis for cross-treatment insights
""")
