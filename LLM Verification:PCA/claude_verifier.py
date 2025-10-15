"""
Standalone Claude Event Verifier

Takes detected_events_toverify.csv and adds verification columns using Claude + web search.
Updates the same CSV file with verification results.

Usage:
    python claude_verifier.py

Make sure you have:
- detected_events_toverify.csv in the same directory
- CLAUDE_API_KEY environment variable set
- anthropic library installed: pip install anthropic
"""

import pandas as pd
import json
import time
import os
from typing import Dict
import anthropic


def verify_single_event_with_claude(event: pd.Series, client: anthropic.Anthropic, model: str) -> Dict:
    """Verify a single detected event using Claude with web search"""

    gvkey = event.get('gvkey', event.get('ticker', 'Unknown'))
    company_name = event.get('company_name', 'Unknown Company')
    event_type = event['event_type']
    event_date = pd.to_datetime(event['event_date'])

    prompt = f"""I detected a potential {event_type} event for company "{company_name}" (GVKEY: {gvkey}).

**Detected Information:**
- Company: {company_name}
- GVKEY: {gvkey}
- Event Type: {event_type}
- Detected Date: {event_date.strftime('%Y-%m-%d')}
- Detection Score: {event.get('detection_score', 'N/A')}
- Confidence: {event.get('confidence', 'N/A')}
- Reasons: {event.get('reasons', 'N/A')}

**Your Task:**
1. Search the web to verify what actually happened to "{company_name}" around {event_date.strftime('%B %Y')}
2. Determine if this was a real {event_type} or something else
3. Provide accurate details

**Respond in this EXACT JSON format (no markdown, just JSON):**
{{
    "verified": true or false,
    "actual_event_type": "bankruptcy" | "merger" | "acquisition" | "halt" | "ipo" | "stock_split" | "ticker_change" | "data_error" | "none" | "unknown",
    "actual_date": "YYYY-MM-DD" or null,
    "llm_confidence": "high" | "medium" | "low",
    "details": "Brief explanation of what happened",
    "acquirer": "Company name if merger/acquisition, else null",
    "acquisition_price": price per share if known, else null
}}

If you cannot find information, set verified=false and llm_confidence="low".
Search the web thoroughly before concluding."""

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        response_text = message.content[0].text

        # Extract JSON from response - handle Claude's extra text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            # Find JSON object in response (starts with { and ends with })
            start = response_text.find('{')
            if start != -1:
                # Find matching closing brace
                brace_count = 0
                end = start
                for i, char in enumerate(response_text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                json_str = response_text[start:end]
            else:
                json_str = response_text.strip()

        result = json.loads(json_str)
        return result

    except Exception as e:
        return {
            'verified': False,
            'actual_event_type': 'error',
            'llm_confidence': 'low',
            'details': f'Error during verification: {str(e)}',
            'error': str(e)
        }


def main():
    """Main verification function"""

    # Check for API key
    api_key = "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"
    if not api_key:
        print("❌ Error: Please set CLAUDE_API_KEY environment variable")
        print("   export CLAUDE_API_KEY=sk-ant-...")
        return

    # Load CSV
    csv_file = 'detected_events_to_verify.csv'
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found in current directory")
        return

    print(f"📁 Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} events")

    # Filter to events with company names
    events_with_names = df[
        (df['company_name'].notna()) &
        (df['company_name'] != 'Unknown') &
        (df['company_name'] != '')
    ].copy()

    if len(events_with_names) == 0:
        print("❌ No events with valid company names found")
        return

    print(f"✓ Found {len(events_with_names)} events with company names to verify")

    # Show cost estimate
    model = "claude-3-5-haiku-20241022"
    cost_per = 0.0003
    estimated_cost = len(events_with_names) * cost_per
    print(f"💰 Estimated cost: ${estimated_cost:.2f}")
    print(f"⏱️  Estimated time: {len(events_with_names) * 0.5 / 60:.1f} minutes")

    proceed = input("\n🤔 Proceed with verification? (yes/no): ")
    if proceed.lower() != 'yes':
        print("❌ Cancelled")
        return

    # Initialize Claude
    client = anthropic.Anthropic(api_key=api_key)

    # Add verification columns if they don't exist
    verification_cols = ['verified', 'actual_event_type', 'actual_date', 'llm_confidence', 'details', 'acquirer', 'acquisition_price']
    for col in verification_cols:
        if col not in df.columns:
            df[col] = None

    # Verify each event
    print("\n🔍 Starting verification...")
    save_interval = 10  # Save every 10 events

    for i, (idx, event) in enumerate(events_with_names.iterrows()):
        company = event.get('company_name', 'Unknown')
        gvkey = event.get('gvkey', event.get('ticker', 'Unknown'))

        print(f"[{i+1}/{len(events_with_names)}] Verifying {company} ({gvkey}) - {event['event_type']}...")

        # Verify with Claude
        result = verify_single_event_with_claude(event, client, model)

        # Update DataFrame
        for col in verification_cols:
            if col in result:
                df.loc[idx, col] = result[col]

        # Show result
        if result.get('verified'):
            print(f"  ✅ Verified as {result.get('actual_event_type')} (confidence: {result.get('llm_confidence')})")
        else:
            print(f"  ❌ Could not verify (confidence: {result.get('llm_confidence')})")

        # Save progress every 10 events
        if (i + 1) % save_interval == 0:
            df.to_csv(csv_file, index=False)
            print(f"  💾 Progress saved ({i+1}/{len(events_with_names)} completed)")

        time.sleep(0.5)  # Rate limiting

    # Handle events without company names
    events_without_names = df[
        (df['company_name'].isna()) |
        (df['company_name'] == 'Unknown') |
        (df['company_name'] == '')
    ]

    if len(events_without_names) > 0:
        print(f"\n⚠️  Marking {len(events_without_names)} events without company names as skipped...")
        for idx in events_without_names.index:
            df.loc[idx, 'verified'] = False
            df.loc[idx, 'actual_event_type'] = 'skipped'
            df.loc[idx, 'llm_confidence'] = 'low'
            df.loc[idx, 'details'] = 'Skipped - no company name for web search'

    # Save updated CSV
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Updated {csv_file} with verification results")

    # Summary
    verified_count = df['verified'].sum() if 'verified' in df.columns else 0
    print(f"\n📊 SUMMARY:")
    print(f"   Total events: {len(df)}")
    print(f"   Verified: {verified_count}")
    print(f"   Not verified: {len(df) - verified_count}")

    if verified_count > 0:
        print(f"\n📈 Verified event types:")
        verified_events = df[df['verified'] == True]
        for event_type, count in verified_events['actual_event_type'].value_counts().items():
            print(f"   {event_type}: {count}")

    print(f"\n✅ Done! Check {csv_file} for results")


if __name__ == "__main__":
    main()

