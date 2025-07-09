categories = [
    "ORDER", "SHIPPING", "CANCEL", "INVOICE",
    "PAYMENT", "REFUND", "FEEDBACK", "CONTACT",
    "ACCOUNT", "DELIVERY", "SUBSCRIPTION"
]

category_entities = {
    "ACCOUNT": ["create_account", "delete_account", "edit_account", "switch_account"],
    "CANCELLATION_FEE": ["check_cancellation_fee"],
    "DELIVERY": ["delivery_options"],
    "FEEDBACK": ["complaint", "review"],
    "INVOICE": ["check_invoice", "get_invoice"],
    "NEWSLETTER": ["newsletter_subscription"],
    "ORDER": ["cancel_order", "change_order", "place_order"],
    "PAYMENT": ["check_payment_methods", "payment_issue"],
    "REFUND": ["check_refund_policy", "track_refund"],
    "SHIPPING_ADDRESS": ["change_shipping_address", "set_up_shipping_address"]
}

entity_to_intents = {
    "Order Number": [
        "cancel_order", "change_order", "change_shipping_address", "check_invoice",
        "check_refund_policy", "complaint", "delivery_options", "delivery_period",
        "get_invoice", "get_refund", "place_order", "track_order", "track_refund"
    ],
    "Invoice Number": ["check_invoice", "get_invoice"],
    "Online Order Interaction": [
        "cancel_order", "change_order", "check_refund_policy", "delivery_period",
        "get_refund", "review", "track_order", "track_refund"
    ],
    "Online Payment Interaction": ["cancel_order", "check_payment_methods"],
    "Online Navigation Step": ["complaint", "delivery_options"],
    "Online Customer Support Channel": [
        "check_refund_policy", "complaint", "contact_human_agent", "delete_account",
        "delivery_options", "edit_account", "get_refund", "payment_issue",
        "registration_problems", "switch_account"
    ],
    "Profile": ["switch_account"],
    "Profile Type": ["switch_account"],
    "Settings": [
        "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
        "check_invoice", "check_payment_methods", "contact_human_agent", "delete_account",
        "delivery_options", "edit_account", "get_invoice", "newsletter_subscription",
        "payment_issue", "place_order", "recover_password", "registration_problems",
        "set_up_shipping_address", "switch_account", "track_order", "track_refund"
    ],
    "Online Company Portal Info": ["cancel_order", "edit_account"],
    "Date": ["check_invoice", "check_refund_policy", "get_refund", "track_order", "track_refund"],
    "Date Range": ["check_cancellation_fee", "check_invoice", "get_invoice"],
    "Shipping Cut-off Time": ["delivery_options"],
    "Delivery City": ["delivery_options"],
    "Delivery Country": ["check_payment_methods", "check_refund_policy", "delivery_options", "review", "switch_account"],
    "Salutation": [
        "cancel_order", "check_payment_methods", "check_refund_policy", "create_account",
        "delete_account", "delivery_options", "get_refund", "recover_password", "review",
        "set_up_shipping_address", "switch_account", "track_refund"
    ],
    "Client First Name": ["check_invoice", "get_invoice"],
    "Client Last Name": ["check_invoice", "create_account", "get_invoice"],
    "Customer Support Phone Number": [
        "change_shipping_address", "contact_customer_service", "contact_human_agent", "payment_issue"
    ],
    "Customer Support Email": [
        "cancel_order", "change_shipping_address", "check_invoice", "check_refund_policy",
        "complaint", "contact_customer_service", "contact_human_agent", "get_invoice",
        "get_refund", "newsletter_subscription", "payment_issue", "recover_password",
        "registration_problems", "review", "set_up_shipping_address", "switch_account"
    ],
    "Live Chat Support": [
        "check_refund_policy", "complaint", "contact_human_agent", "delete_account",
        "delivery_options", "edit_account", "get_refund", "payment_issue", "recover_password",
        "registration_problems", "review", "set_up_shipping_address", "switch_account", "track_order"
    ],
    "Website URL": [
        "check_payment_methods", "check_refund_policy", "complaint", "contact_customer_service",
        "contact_human_agent", "create_account", "delete_account", "delivery_options",
        "get_refund", "newsletter_subscription", "payment_issue", "place_order", "recover_password",
        "registration_problems", "review", "switch_account"
    ],
    "Upgrade Account": ["create_account", "edit_account", "switch_account"],
    "Account Type": [
        "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
        "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
        "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
        "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
        "newsletter_subscription", "payment_issue", "place_order", "recover_password",
        "registration_problems", "review", "set_up_shipping_address", "switch_account",
        "track_order", "track_refund"
    ],
    "Account Category": [
        "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
        "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
        "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
        "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
        "newsletter_subscription", "payment_issue", "place_order", "recover_password",
        "registration_problems", "review", "set_up_shipping_address", "switch_account",
        "track_order", "track_refund"
    ],
    "Account Change": ["switch_account"],
    "Program": ["place_order"],
    "Refund Amount": ["track_refund"],
    "Money Amount": ["check_refund_policy", "complaint", "get_refund", "track_refund"],
    "Store Location": ["complaint", "delivery_options", "place_order"]
}